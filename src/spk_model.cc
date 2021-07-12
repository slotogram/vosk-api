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

#include "spk_model.h"

SpkModel::SpkModel(const char *speaker_path) {
    std::string speaker_path_str(speaker_path);

    ReadConfigFromFile(speaker_path_str + "/mfcc.conf", &spkvector_mfcc_opts);
    spkvector_mfcc_opts.frame_opts.allow_downsample = true; // It is safe to downsample

    ReadKaldiObject(speaker_path_str + "/final.ext.raw", &speaker_nnet);
    SetBatchnormTestMode(true, &speaker_nnet);
    SetDropoutTestMode(true, &speaker_nnet);
    CollapseModel(nnet3::CollapseModelConfig(), &speaker_nnet);
	//this speeds up multiple consequent xvector computations
	NnetSimpleComputationOptions opts;
	nnet3::CachingOptimizingCompilerOptions compiler_config;
	compiler  = new nnet3::CachingOptimizingCompiler (speaker_nnet, opts.optimize_config, compiler_config);

    ReadKaldiObject(speaker_path_str + "/mean.vec", &mean);
    ReadKaldiObject(speaker_path_str + "/transform.mat", &transform);
	ReadKaldiObject(speaker_path_str + "/plda", &plda);

    ref_cnt_ = 1;
}

kaldi::Vector<BaseFloat> SpkModel::LoadXVector(std::string path, std::string key)
{
	Vector<BaseFloat> ivector;
	SequentialBaseFloatVectorReader train_ivector_reader(path);
	for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
		std::string spk = train_ivector_reader.Key();
		ivector = train_ivector_reader.Value();
	}
	return ivector;
}

bool SpkModel::SaveXVector(kaldi::Vector<BaseFloat> xvector, std::string path, std::string key)
{
	BaseFloatVectorWriter vector_writer(path);

	vector_writer.Write(key, xvector);
	return true;
}

void SpkModel::Ref()
{
    std::atomic_fetch_add_explicit(&ref_cnt_, 1, std::memory_order_relaxed);
}

void SpkModel::Unref()
{
    if (std::atomic_fetch_sub_explicit(&ref_cnt_, 1, std::memory_order_release) == 1) {
         std::atomic_thread_fence(std::memory_order_acquire);
         delete compiler;
		 delete this;
    }
}
