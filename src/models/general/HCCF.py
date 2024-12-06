from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from models.BaseModel import GeneralModel
# from Params import args
# from Utils.Utils import pairPredict, contrastLoss

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class HCCF(GeneralModel):
	reader = 'BaseReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'hyper_num','leaky']

	@staticmethod
	def parse_model_args(parser):
		parser.add_argument('--emb_size', type=int, default=32,
                            help='Size of embedding vectors.')
		parser.add_argument('--hyper_num', type=int, default=128,
                            help='number of hyperedges')
		parser.add_argument('--leaky', default=0.5, type=float, help='slope of leaky relu')
		parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
		return GeneralModel.parse_model_args(parser)
	
	# @staticmethod
	# def init_weights(m):
	# 	if 'Parameter' in str(type(m)):
	# 		nn.init.xavier_normal_(m.weight.data)
	# 		if m.bias is not None:
	# 			nn.init.normal_(m.bias.data)
	# 	elif 'Embedding' in str(type(m)):
	# 		nn.init.xavier_normal_(m.weight.data)

	def __init__(self,args, corpus):
		super().__init__(args,corpus)
		self.emb_size = args.emb_size
		self.hyper_num = args.hyper_num
		self.leaky = args.leaky
		self.gnn_layer = args.gnn_layer
		self._define_params()


	def _define_params(self):
		self.uEmbeds = nn.Parameter(init(t.empty(self.user_num, self.emb_size)))
		self.iEmbeds = nn.Parameter(init(t.empty(self.item_num, self.emb_size)))
		self.gcnLayer = GCNLayer(self.leaky)
		self.hgnnLayer = HGNNLayer(self.leaky)
		self.uHyper = nn.Parameter(init(t.empty(self.emb_size, self.hyper_num)))
		self.iHyper = nn.Parameter(init(t.empty(self.emb_size, self.hyper_num)))
		self.edgeDropper = SpAdjDropEdge()

	def forward(self,feed_dict):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper
		
	def old_forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper

		for i in range(self.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keepRate), lats[-1][:self.user_num])
			hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keepRate), lats[-1][self.user_num:])
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			lats.append(temEmbeds + hyperLats[-1])
		embeds = sum(lats)
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:self.user_num], embeds[self.user_num:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:self.user_num], embeds2[:self.user_num], t.unique(ancs), args.temp) + contrastLoss(embeds1[self.user_num:], embeds2[self.user_num:], t.unique(poss), args.temp)
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 1.0)
		return embeds[:self.user_num], embeds[self.user_num:]

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
