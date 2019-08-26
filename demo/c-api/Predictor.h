/*!
 * Copyright 2019
 *
 * Create time 2019-08-24 pm 14:23
 * by zrb
 */

#ifndef KALDI_PREDICTOR_H
#define KALDI_PREDICTOR_H

#include <stdio.h>
#include <stdlib.h>
#include <xgboost/c_api.h>
#include <vector>
#include <sys/time.h>
#include <fstream>
#include<pthread.h>

#define MODEL_NUM 2
#define MISS_DEFAULT -1
#define safe_xgboost(call) {                                            \
int err = (call);                                                       \
if (err != 0) {                                                         \
  fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
  return false;                                                              \
}                                                                       \
}

namespace kaldi {

class Predictor {
public:
	Predictor();
	~Predictor();
	bool Init(const std::vector<string> modelfile, const std::vector<string> featmapfd);

	bool Run(std::vector<BaseFloat> fea_set, std::vector<BaseFloat>& result);

private:
  bool  getPredictFeatures(std::vector<BaseFloat> fea_set,std::vector<int> & featMap, DMatrixHandle * dataHandle);
  DMatrixHandle mDataHandle;
  BoosterHandle mPronBstHandle;
  BoosterHandle mFluBstHandle;
  std::vector<int> mPronFeatId;
  std::vector<int> mFluFeatId;
  static const pthread_mutex_t xgb_mutex = PTHREAD_MUTEX_INITIALIZER;

};

}  // End namespace kaldi

#endif
