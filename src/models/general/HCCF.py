from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
import numpy as np
import scipy.sparse as sp
# from Params import args
from utils.utils import pairPredict, contrastLoss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class HCCF(GeneralModel):
	reader = 'BaseReader'
	runner = 'HCCFRunner'
	extra_log_args = ['emb_size', 'hyper_num','leaky','gnn_layer']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
		parser.add_argument('--hyper_num', type=int, default=128,
                            help='number of hyperedges')
		parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
		parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
		return GeneralModel.parse_model_args(parser)
	
	@staticmethod
	def normalizeAdj(mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()
	
	@staticmethod
	def build_adjmat(data,user,item):
		mat=sp.coo_matrix((np.ones_like(data['user_id']), (data['user_id'], data['item_id'])))
		a = sp.csr_matrix((user, user))
		b = sp.csr_matrix((item, item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# mat = (mat + sp.eye(mat.shape[0])) * 1.0
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		mat=mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def __init__(self,args, corpus):
		super().__init__(args,corpus)
		self.emb_size = args.emb_size
		self.hyper_num = args.hyper_num
		self.leaky = args.leaky
		self.gnn_layer = args.gnn_layer
		self.adj= self.build_adjmat(corpus.data_df['train'],self.user_num,self.item_num)
		self.keepRate = 0.5
		self.temp=1
		self.reg=1e-7
		self.ssl_reg=1e-3
		print(self.user_num,self.item_num)
		#self.optimizer=t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		self._define_params()


	def _define_params(self):
		self.uEmbeds = nn.Parameter(init(t.empty(self.user_num, self.emb_size)))
		self.iEmbeds = nn.Parameter(init(t.empty(self.item_num, self.emb_size)))
		self.gcnLayer = GCNLayer(self.leaky)
		self.hgnnLayer = HGNNLayer(self.leaky)
		self.uHyper = nn.Parameter(init(t.empty(self.emb_size, self.hyper_num)))
		self.iHyper = nn.Parameter(init(t.empty(self.emb_size, self.hyper_num)))
		self.edgeDropper = SpAdjDropEdge()
	
	def predict(self,batch):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper
		for i in range(self.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(self.adj, self.keepRate), lats[-1])
			hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-self.keepRate), lats[-1][:self.user_num])
			hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-self.keepRate), lats[-1][self.user_num:])
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			lats.append(temEmbeds + hyperLats[-1])
		embeds = sum(lats)
		uEmbeds, iEmbeds = embeds[:self.user_num], embeds[self.user_num:]
		m1=uEmbeds[batch['user_id']]
		m2=iEmbeds[batch['item_id']]
		allPreds=(m1[:, None, :] * m2).sum(dim=-1)
		#print(allPreds.shape)
		#allPreds = #* (1 - batch['trnmask']) - batch['trnmask'] * 1e8

		return allPreds

	def forward(self,feed_dict):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper
		for i in range(self.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(self.adj, self.keepRate), lats[-1])
			hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-self.keepRate), lats[-1][:self.user_num])
			hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-self.keepRate), lats[-1][self.user_num:])
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			lats.append(temEmbeds + hyperLats[-1])
		embeds = sum(lats)
		uEmbeds, iEmbeds = embeds[:self.user_num], embeds[self.user_num:]
		ancEmbeds = uEmbeds[feed_dict['ancs']]
		posEmbeds = iEmbeds[feed_dict['poss']]
		negEmbeds = iEmbeds[feed_dict['negs']]
		return {"ancEmbeds":ancEmbeds,"posEmbeds":posEmbeds,"negEmbeds":negEmbeds,'gnnLats':gnnLats,'hyperLats':hyperLats}

	def calcRegLoss(self):
		ret = 0
		for W in self.parameters():
			ret += W.norm(2).square()
		# ret += (model.usrStruct + model.itmStruct)
		return ret
	
	def loss(self, output,feed_dict):
		ancEmbeds = output['ancEmbeds']
		posEmbeds = output['posEmbeds']
		negEmbeds = output['negEmbeds']
		gcnEmbedsLst = output['gnnLats']
		hyperEmbedsLst = output['hyperLats']
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		for i in range(self.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:self.user_num], embeds2[:self.user_num], t.unique(feed_dict['ancs']), self.temp) + contrastLoss(embeds1[self.user_num:], embeds2[self.user_num:], t.unique(feed_dict['poss']), self.temp)
		sslLoss = sslLoss *self.ssl_reg
		regLoss = self.calcRegLoss() * self.reg
		loss = bprLoss + regLoss + sslLoss
		return loss
	
	class Dataset(GeneralModel.Dataset):#就是torch的Dataset类
		def __init__(self, model, corpus, phase: str):
			super().__init__(model, corpus, phase)
			if self.phase=='train':
				coomat=sp.coo_matrix((np.ones_like(self.data['user_id']), (self.data['user_id'], self.data['item_id'])))
				self.rows=coomat.row
				self.cols=coomat.col
				self.dokmat=coomat.todok()
				self.negs = np.zeros(len(self.rows)).astype(np.int32)
			else:
				trnmat=sp.coo_matrix((np.ones_like(self.corpus.data_df['train']['user_id']), (self.corpus.data_df['train']['user_id'], self.corpus.data_df['train']['item_id'])))
				self.csrmat = (trnmat.tocsr() != 0) * 1.0
				coomat=sp.coo_matrix((np.ones_like(self.data['user_id']), (self.data['user_id'], self.data['item_id'])))
				tstLocs = [None] * coomat.shape[0]
				tstUsrs = set()
				for i in range(len(coomat.data)):
					row = coomat.row[i]
					col = coomat.col[i]
					if tstLocs[row] is None:
						tstLocs[row] = list()
					tstLocs[row].append(col)
					tstUsrs.add(row)
				tstUsrs = np.array(list(tstUsrs))
				self.tstUsrs = tstUsrs#似乎就是找出所有用户，并去重
				self.tstLocs = tstLocs

		def __len__(self):
			if (self.phase=='train'):
				return len(self.rows)
			return len(self.tstUsrs)

		def __getitem__(self, index: int) -> dict:
			return self._get_feed_dict(index)

		# ! Key method to construct input data for a single instance
		def _get_feed_dict(self, index: int) -> dict:
			if (self.phase=='train'):
				return {'ancs': self.rows[index], 'poss': self.cols[index], 'negs': self.negs[index]}
			else:
				user_id, target_item = self.data['user_id'][index], self.data['item_id'][index]
				neg_items = self.data['neg_items'][index]
				item_ids = np.concatenate([[target_item], neg_items]).astype(int)
				return {'usr': self.tstUsrs[index], 'trnmask': np.reshape(self.csrmat[self.tstUsrs[index]].toarray(), [-1]),'user_id':user_id,'item_id':item_ids}


		# Called before each training epoch (only for the training dataset)
		def actions_before_epoch(self):
			for i in range(len(self.rows)):
				u = self.rows[i]
				while True:
					iNeg = np.random.randint(self.item_num)
					if (u, iNeg) not in self.dokmat:
						break
				self.negs[i] = iNeg


			


class GCNLayer(nn.Module):
	def __init__(self,leaky=0.5):
		super(GCNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=leaky)

	def forward(self, adj, embeds):
		return (t.spmm(adj, embeds))

class HGNNLayer(nn.Module):
	def __init__(self, leaky=0.5):
		super(HGNNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=leaky)
	
	def forward(self, adj, embeds):
		# lat = self.act(adj.T @ embeds)
		# ret = self.act(adj @ lat)
		lat = (adj.T @ embeds)
		ret = (adj @ lat)
		return ret

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
