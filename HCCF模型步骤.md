# HCCF算法流程
1. 读入数据

   数据好像就是两个矩阵，一个训练矩阵，一个测试矩阵，然后矩阵n x m，n代表用户数量，m代表物品数量

   ```python
   	handler = DataHandler()
   	handler.LoadData()
   ```

   ```python
   from scipy.sparse import csr_matrix, coo_matrix, dok_matrix #设计scipy对稀疏矩阵的三种处理方式，略
   class TrnData(data.Dataset):
   	def __init__(self, coomat):
   		self.rows = coomat.row #这里rows和cols的数量是相等的，代表所有交互的结点
   		self.cols = coomat.col
   		self.dokmat = coomat.todok()#转换成字典形式
   		self.negs = np.zeros(len(self.rows)).astype(np.int32)
   
   	def negSampling(self):#随机采样一些出来
   		for i in range(len(self.rows)):
   			u = self.rows[i]
   			while True:
   				iNeg = np.random.randint(args.item)
   				if (u, iNeg) not in self.dokmat:
   					break
   			self.negs[i] = iNeg
   
   	def __len__(self):
   		return len(self.rows)
   
   	def __getitem__(self, idx):
   		return self.rows[idx], self.cols[idx], self.negs[idx]
   ```

   

2. 准备模型
3. 训练（不重要了）

# 框架算法流程

1. 确定GPU
2. 利用所选模型的reader读入数据。
3. 定义所选模型，将其放到gpu上
4. 定义所选模型的数据集，这里用到了之前reader读出来的数据
5. 运行模型（训练），这里用到了该模型的runner
6. 评估模型在验证集和测试集上的结果



**一些参考**

+ generalModel的基本参数

```python
#generalModel自带的参数
  def __init__(self, args, corpus):
		super().__init__(args, corpus)
		self.user_num = corpus.n_users
		self.item_num = corpus.n_items
		self.num_neg = args.num_neg
		self.dropout = args.dropout
		self.test_all = args.test_all
```

+ BaseReader的基本参数：关键是这个train_clicked_set和residual_clicked_set，以及train dev test的self.data_df(字典)

```python
  class BaseReader(object):
      @staticmethod
      def parse_data_args(parser):
          parser.add_argument('--path', type=str, default='../data',
                              help='Input data dir.')
          parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                              help='Choose a dataset.')
          parser.add_argument('--sep', type=str, default='\t',
                              help='sep of csv file.')
          return parser
  
      def __init__(self, args):
          self.sep = args.sep
          self.prefix = args.path
          self.dataset = args.dataset
          self._read_data()
  
          self.train_clicked_set = dict()  # store the clicked item set of each user in training set
          self.residual_clicked_set = dict()  # store the residual clicked item set of each user
          for key in ['train', 'dev', 'test']:
              df = self.data_df[key]
              for uid, iid in zip(df['user_id'], df['item_id']):
                  if uid not in self.train_clicked_set:
                      self.train_clicked_set[uid] = set()
                      self.residual_clicked_set[uid] = set()
                  if key == 'train':#区分用户是否在训练集中出现过
                      self.train_clicked_set[uid].add(iid)
                  else:
                      self.residual_clicked_set[uid].add(iid)
  
      def _read_data(self):
          logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
          self.data_df = dict()
          print(self.prefix)
          for key in ['train', 'dev', 'test']:
              self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'), sep=self.sep).reset_index(drop=True).sort_values(by = ['user_id','time'])
              self.data_df[key] = utils.eval_list_columns(self.data_df[key])
  
          logging.info('Counting dataset statistics...')
          key_columns = ['user_id','item_id','time']
          if 'label' in self.data_df['train'].columns: # Add label for CTR prediction,没有label就不用
              key_columns.append('label')
          self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
          self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
          for key in ['dev', 'test']:
              if 'neg_items' in self.data_df[key]:
                  neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                  assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
          logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
              self.n_users - 1, self.n_items - 1, len(self.all_df)))
          if 'label' in key_columns:
              positive_num = (self.all_df.label==1).sum()
              logging.info('"# positive interaction": {} ({:.1f}%)'.format(
  				positive_num, positive_num/self.all_df.shape[0]*100))
```

+ Reader得到的data_df是如何传到Dataset中的:用这个corpus和phase

```python 
	class Dataset(BaseDataset):#就是torch的Dataset类
		def __init__(self, model, corpus, phase: str):
			self.model = model  # model object reference
			self.corpus = corpus  # reader object reference
			self.phase = phase  # train / dev / test

			self.buffer_dict = dict()
			#self.data = utils.df_to_dict(corpus.data_df[phase])#this raise the VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences warning
			self.data = corpus.data_df[phase].to_dict('list')
			# ↑ DataFrame is not compatible with multi-thread operations
```



+ reader之后的数据

```python
1. data_df[][],两维度，第一个是['train','dev','test']，第二个由具体数据确定，可能是['user_id','item_id','neg_id']有reader定义
2. data[]，一个维度，指带具体数据类型（训练测试验证）的，某个key，由模型的Dataset确定。

```

+ 矩阵的存储形式

```python
1. coo_matrix，普通的稀疏矩阵，可以通过行号、列号以及对应值创建
# 非零元素的行索引
row = np.array([0,0,0,0,2,2])
# 非零元素的列索引
col = np.array([0,1,2,3,4,3])
# 非零元素的值
data = np.array([4, 5, 6, 7,7,7])

# 创建 COO 矩阵
coo = sp.coo_matrix((data, (row, col)), shape=(6, 6))

2、dok_matrix， 转换为字典，key为（行，列）元组，值为value
dok_matrix =coo.todok()

3.csr_matrix 压缩稀疏行，又数值，列索引，每行第一个非零元素的索引构成
csr_mat = coo.tocsr()
# 打印 CSR 矩阵
print(csr_mat)
print("CSR matrix data:", csr_mat.data)
print("CSR matrix indices:", csr_mat.indices)#列索引
print("CSR matrix indptr:", csr_mat.indptr)#每行第一个非零元素的索引
```

