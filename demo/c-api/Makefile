SRC=c-api-demo.c
TGT=c-api-demo
#SRC=lr-demo.c
#TGT=lr-demo
#SRC=audio-predict.c
#TGT=audio-predict
#SRC=zrb-demo.c
#TGT=zrb_demo


#cc=cc
cc=g++
CFLAGS ?=-O3
XGBOOST_ROOT ?=../..
INCLUDE_DIR=-I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include
#LIB_DIR=-L$(XGBOOST_ROOT)/lib
STAT_LIB=$(XGBOOST_ROOT)/lib/libxgboost.a $(XGBOOST_ROOT)/dmlc-core/libdmlc.a $(XGBOOST_ROOT)/rabit/lib/librabit.a  
build: $(TGT)

$(TGT): $(SRC) Makefile
	#$(cc) $(CFLAGS) $(INCLUDE_DIR) $(LIB_DIR) -o $(TGT) $(SRC) -lxgboost
	$(cc) $(CFLAGS) $(INCLUDE_DIR) -o $(TGT) $(SRC) $(STAT_LIB) -pthread -lm  -fopenmp -lrt -lrt

run: $(TGT)
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib ./$(TGT)

clean:
	rm -f $(TGT)
