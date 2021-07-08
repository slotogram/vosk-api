// Copyright 2020 Alpha Cephei Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef VOSK_SPK_MODEL_H
#define VOSK_SPK_MODEL_H

#include "base/kaldi-common.h"
#include "online2/online-feature-pipeline.h"
#include "nnet3/nnet-utils.h"
#include "ivector/plda.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"

using namespace kaldi;
using namespace kaldi::nnet3;
class KaldiRecognizer;

class SpkModel {

public:
    SpkModel(const char *spk_path);
	kaldi::Vector<BaseFloat> LoadXVector(std::string path, std::string key);
	bool SaveXVector(kaldi::Vector<BaseFloat> xvector, std::string path, std::string key);
    void Ref();
    void Unref();

protected:
    friend class KaldiRecognizer;
	~SpkModel() { 		}; //compiler->~CachingOptimizingCompiler();

    kaldi::nnet3::Nnet speaker_nnet;
    kaldi::Vector<BaseFloat> mean;
    kaldi::Matrix<BaseFloat> transform;
	Plda plda;
	nnet3::CachingOptimizingCompiler *compiler;

    MfccOptions spkvector_mfcc_opts;

    int ref_cnt_;
};

#endif /* VOSK_SPK_MODEL_H */
