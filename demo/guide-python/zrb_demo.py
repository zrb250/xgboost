#!/usr/bin/python
import xgboost as xgb

### load data in do training
#dtrain = xgb.DMatrix('../data/agaricus.txt.train')
#dtest = xgb.DMatrix('../data/agaricus.txt.test')
dtrain = xgb.DMatrix('../data/audio_train.txt')
dtest = xgb.DMatrix('../data/audio_test.txt')
param = {'max_depth':2, 
        'eta':0.5,
        'gamma':0.1,
        'min_child_weight':5,
        'silent':1, 
        'objective':'multi:softmax',
        'num_class':17
        }
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
num_round = 3
bst = xgb.train(param, dtrain, num_round, watchlist)
bst.save_model("./xgb.model");
#bst.dump_model('dump.raw.txt');
bst.dump_model('dump.raw.txt','featmap.txt');
print ('start testing predict the leaf indices')
### predict using first 2 tree
leafindex = bst.predict(dtest, ntree_limit=2, pred_leaf=True)
print(leafindex.shape)
print(leafindex)
### predict all trees
leafindex = bst.predict(dtest, pred_leaf=True)
print(leafindex.shape)
