/*!
 * Copyright 2019
 *
 * Create time 2019-08-24 pm 14:23
 * by zrb
 */
#include "Predictor.h"

using namespace std;
namespace kaldi {

Predictor::Predictor(){}

Predictor::~Predictor(){
    XGDMatrixFree(mPronDataHandle);
    XGDMatrixFree(mFluDataHandle);
    XGBoosterFree(mPronBstHandle);
    XGBoosterFree(mFluBstHandle);
}

bool Predictor::Init(const std::vector<string>& modelfile, const std::vector<string>& featmapfd){

  if((modelfile.size() != MODEL_NUM) || (featmapfd.size() != MODEL_NUM)){
      printf("param error:Predictor::Init");
      return false;
  }
     //init PronBstHandle
    safe_xgboost(XGBoosterCreate(0, 0, &mPronBstHandle));
    safe_xgboost(XGBoosterLoadModel(mPronBstHandle,modelfile[0]));
    ifstream finPron(featmapfd[0]);
    if(!finPron){
      printf("open featmap fail:%s", featmapfd[0]);
      return false;
    }
    while(getline(finPron,s)){
    	 mPronFeatId.push_back(atoi(s.c_str()))
    	}

    //init  FluBstHandle
    safe_xgboost(XGBoosterCreate(0, 0, &mFluBstHandle));
    safe_xgboost(XGBoosterLoadModel(mFluBstHandle,modelfile[1]));
    ifstream finFlu(featmapfd[1]);
    if(!finFlu){
          printf("open featmap fail:%s", featmapfd[1]);
          return false;
    }
    while(getline(finPron,s)){
        	 mFluFeatId.push_back(atoi(s.c_str()))
     }

  return true;
}

bool Predictor::Run(std::vector<BaseFloat> fea_set, std::vector<float>& result){

  if((fea_set.size() < mPronFeatId.size()) || (fea_set.size() < mFluFeatId.size())){
      printf("fea_set lenght invaid!");
      return false;
  }
  float pronFeature[mPronFeatId.size()];
  float fluFeatrue[mFluFeatId.size()];

  if(!getPredictFeatures(fea_set, mFluFeatId, &mFluDataHandle)){
     printf("getPredictFeatures  fluFeatrue error!");
     return false;
  }

  if(!getPredictFeatures(fea_set, mPronFeatId, &mPronDataHandle)){
      printf("getPredictFeatures  pronFeature error!");
      return false;
  }

  bst_ulong pronLen = 0;
  const float* pronResult = NULL;
  bst_ulong fluLen = 0;
  const float* fluResult = NULL;

  //xgb 线程不安全，预测环节加锁
  pthread_mutex_lock(&Predictor::xgb_mutex);
  safe_xgboost(XGBoosterPredict(mFluBstHandle, mFluDataHandle, 0, 0, &fluLen, &fluResult));
  safe_xgboost(XGBoosterPredict(mPronBstHandle, mPronDataHandle, 0, 0, &pronLen, &pronResult));
  pthread_mutex_unlock(&Predictor::xgb_mutex);

  result.push_back(pronResult[0]);
  result.push_back(fluResult[0]);

  return true;
}

bool Predictor::getPredictFeatures(std::vector<BaseFloat> fea_set,std::vector<int> & featMap, DMatrixHandle * dataHandle){

  float features[featMap.size()];
  for(int i = 0; i < featMap.size(); i++){
    features[i] = fea_set[featMap.[i]];
  }

 int ret = XGDMatrixCreateFromMat(out, 1, input.size(), MISS_DEFAULT, dataHandle);

 return ret != 0 ? false : true;
}

} // End namespace kaldi
