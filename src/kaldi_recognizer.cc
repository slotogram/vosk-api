// Copyright 2019-2020 Alpha Cephei Inc.
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

#include "kaldi_recognizer.h"
#include "json.h"
#include "fstext/fstext-utils.h"
#include "lat/sausages.h"
#include "language_model.h"
#include "ivector/voice-activity-detection.h"
#include "online/online-audio-source.h"
#include "online2/online-endpoint.h"
#include <string.h>
#include <experimental/filesystem>

using namespace fst;
using namespace kaldi::nnet3;

KaldiRecognizer::KaldiRecognizer(Model *model, float sample_frequency) : model_(model), spk_model_(0), sample_frequency_(sample_frequency) {

    model_->Ref();

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    if (!model_->hclg_fst_) {
        if (model_->hcl_fst_ && model_->g_fst_) {
            decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *model_->g_fst_, model_->disambig_);
        } else {
            KALDI_ERR << "Can't create decoding graph";
        }
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    InitState();
    InitRescoring();
}

KaldiRecognizer::KaldiRecognizer(Model *model, float sample_frequency, char const *grammar) : model_(model), spk_model_(0), sample_frequency_(sample_frequency)
{
    model_->Ref();

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    if (model_->hcl_fst_) {
        json::JSON obj;
        obj = json::JSON::Load(grammar);

        if (obj.length() <= 0) {
            KALDI_WARN << "Expecting array of strings, got: '" << grammar << "'";
        } else {
            KALDI_LOG << obj;

            LanguageModelOptions opts;

            opts.ngram_order = 2;
            opts.discount = 0.5;

            LanguageModelEstimator estimator(opts);
            for (int i = 0; i < obj.length(); i++) {
                bool ok;
                string line = obj[i].ToString(ok);
                if (!ok) {
                    KALDI_ERR << "Expecting array of strings, got: '" << obj << "'";
                }

                std::vector<int32> sentence;
                stringstream ss(line);
                string token;
                while (getline(ss, token, ' ')) {
                    int32 id = model_->word_syms_->Find(token);
                    if (id == kNoSymbol) {
                        KALDI_WARN << "Ignoring word missing in vocabulary: '" << token << "'";
                    } else {
                        sentence.push_back(id);
                    }
                }
                estimator.AddCounts(sentence);
            }
            g_fst_ = new StdVectorFst();
            estimator.Estimate(g_fst_);

            decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *g_fst_, model_->disambig_);
        }
    } else {
        KALDI_WARN << "Runtime graphs are not supported by this model";
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    InitState();
    InitRescoring();
}

KaldiRecognizer::KaldiRecognizer(Model *model, float sample_frequency, SpkModel *spk_model) : model_(model), spk_model_(spk_model), sample_frequency_(sample_frequency) {

    model_->Ref();
    spk_model->Ref();

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    if (!model_->hclg_fst_) {
        if (model_->hcl_fst_ && model_->g_fst_) {
            decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *model_->g_fst_, model_->disambig_);
        } else {
            KALDI_ERR << "Can't create decoding graph";
        }
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);

    InitState();
    InitRescoring();
}

KaldiRecognizer::KaldiRecognizer(SpkModel *spk_model) : model_(0), spk_model_(spk_model) {

	spk_model->Ref();

	spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);
	sample_frequency_ = spk_model_->spkvector_mfcc_opts.frame_opts.samp_freq;
	//InitState();
	//InitRescoring();
}

OnlinePaSource* init_kaldi_portaudio(float kSampleFreq)
{
	//do_endpointing = true;
	//online = true;
	// Timeout interval for the PortAudio source
	const int32 kTimeout = 500; // half second

	// PortAudio's internal ring buffer size in bytes
	const int32 kPaRingSize = 32768;
	// Report interval for PortAudio buffer overflows in number of feat. batches
	const int32 kPaReportInt = 4;
	//OnlinePaSource au_src(kTimeout, kSampleFreq, kPaRingSize, kPaReportInt);

	OnlinePaSource* au_src = new OnlinePaSource(kTimeout, kSampleFreq, kPaRingSize, kPaReportInt);
	return au_src;
}


KaldiRecognizer::KaldiRecognizer(SpkModel *spk_model, bool need_mic) : model_(0), spk_model_(spk_model){

	spk_model->Ref();
	spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);	
	sample_frequency_ = spk_model_->spkvector_mfcc_opts.frame_opts.samp_freq;
	au_src_ = init_kaldi_portaudio(sample_frequency_);
	/*try
	{
		au_src_ = init_kaldi_portaudio(sample_frequency_);
	}
	catch (std::runtime_error err)
	{
		au_src_ = nullptr;
	}*/
}


KaldiRecognizer::~KaldiRecognizer() {
	delete au_src_;
	if (model_) {
	delete decoder_;
    delete feature_pipeline_;
    delete silence_weighting_;
    delete g_fst_;
    delete decode_fst_;
    delete spk_feature_;
    delete lm_fst_;

    delete info;
    delete lm_to_subtract_det_backoff;
    delete lm_to_subtract_det_scale;
    delete lm_to_add_orig;
    delete lm_to_add;

    model_->Unref();
	}
	if (spk_model_)
	{
		spk_model_->Unref();
		delete out_str_;
	}
}

void KaldiRecognizer::InitState()
{
    frame_offset_ = 0;
    samples_processed_ = 0;
    samples_round_start_ = 0;

    state_ = RECOGNIZER_INITIALIZED;
}

void KaldiRecognizer::InitRescoring()
{
    if (model_->rnnlm_lm_fst_) {
        float lm_scale = 0.5;
        int lm_order = 4;

        info = new kaldi::rnnlm::RnnlmComputeStateInfo(model_->rnnlm_compute_opts, model_->rnnlm, model_->word_embedding_mat);
        lm_to_subtract_det_backoff = new fst::BackoffDeterministicOnDemandFst<fst::StdArc>(*model_->rnnlm_lm_fst_);
        lm_to_subtract_det_scale = new fst::ScaleDeterministicOnDemandFst(-lm_scale, lm_to_subtract_det_backoff);
        lm_to_add_orig = new kaldi::rnnlm::KaldiRnnlmDeterministicFst(lm_order, *info);
        lm_to_add = new fst::ScaleDeterministicOnDemandFst(lm_scale, lm_to_add_orig);

    } else if (model_->std_lm_fst_) {
        fst::CacheOptions cache_opts(true, 50000);
        fst::ArcMapFstOptions mapfst_opts(cache_opts);
        fst::StdToLatticeMapper<kaldi::BaseFloat> mapper;
        lm_fst_ = new fst::ArcMapFst<fst::StdArc, kaldi::LatticeArc, fst::StdToLatticeMapper<kaldi::BaseFloat> >(*model_->std_lm_fst_, mapper, mapfst_opts);
    }
}

void KaldiRecognizer::CleanUp()
{
    delete silence_weighting_;
	if (model_) {
		silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

				 
													 

		if (decoder_)
			frame_offset_ += decoder_->NumFramesDecoded();
	}
    // Each 10 minutes we drop the pipeline to save frontend memory in continuous processing
    // here we drop few frames remaining in the feature pipeline but hope it will not
    // cause a huge accuracy drop since it happens not very frequently.

    // Also restart if we retrieved final result already

    if (decoder_ == nullptr || state_ == RECOGNIZER_FINALIZED || frame_offset_ > 20000) {
		if (model_) {			   
        samples_round_start_ += samples_processed_;
        samples_processed_ = 0;
        frame_offset_ = 0;

        delete decoder_;
        delete feature_pipeline_;

        feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
        decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);
		}
        if (spk_model_) {
            delete spk_feature_;
            spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);
        }
    } else {
        decoder_->InitDecoding(frame_offset_);
    }
}

void KaldiRecognizer::UpdateSilenceWeights()
{
    if (silence_weighting_->Active() && feature_pipeline_->NumFramesReady() > 0 &&
        feature_pipeline_->IvectorFeature() != nullptr) {
        vector<pair<int32, BaseFloat> > delta_weights;
        silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
        silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                                          frame_offset_ * 3,
                                          &delta_weights);
        feature_pipeline_->UpdateFrameWeights(delta_weights);
    }
}

void KaldiRecognizer::SetMaxAlternatives(int max_alternatives)
{
    max_alternatives_ = max_alternatives;
}

void KaldiRecognizer::SetWords(bool words)
{
    words_ = words;
}

void KaldiRecognizer::SetSpkModel(SpkModel *spk_model)
{
    if (state_ == RECOGNIZER_RUNNING) {
        KALDI_ERR << "Can't add speaker model to already running recognizer";
        return;
    }
    spk_model_ = spk_model;
    spk_model_->Ref();
    spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);
}

bool KaldiRecognizer::AcceptWaveform(const char *data, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len / 2, kUndefined);
    for (int i = 0; i < len / 2; i++)
        wave(i) = *(((short *)data) + i);
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(const short *sdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = sdata[i];
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(const float *fdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = fdata[i];
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(Vector<BaseFloat> &wdata)
{
    // Cleanup if we finalized previous utterance or the whole feature pipeline
    if (!(state_ == RECOGNIZER_RUNNING || state_ == RECOGNIZER_INITIALIZED)) {
        CleanUp();
    }
    state_ = RECOGNIZER_RUNNING;
	if (model_) {
    int step = static_cast<int>(sample_frequency_ * 0.2);
    for (int i = 0; i < wdata.Dim(); i+= step) {
        SubVector<BaseFloat> r = wdata.Range(i, std::min(step, wdata.Dim() - i));
        feature_pipeline_->AcceptWaveform(sample_frequency_, r);
        UpdateSilenceWeights();
        decoder_->AdvanceDecoding();
    }
    samples_processed_ += wdata.Dim();

    if (spk_feature_) {
        spk_feature_->AcceptWaveform(sample_frequency_, wdata);
    }

    if (decoder_->EndpointDetected(model_->endpoint_config_)) {
        return true;
    }
	}
	if (spk_feature_ && ! model_) {
        spk_feature_->AcceptWaveform(sample_frequency_, wdata);
    }
    return false;
}

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
    const nnet3::Nnet &nnet, nnet3::CachingOptimizingCompiler *compiler,
    Vector<BaseFloat> *xvector) 
{
    nnet3::ComputationRequest request;
    request.need_model_derivative = false;
    request.store_component_stats = false;
    request.inputs.push_back(
    nnet3::IoSpecification("input", 0, features.NumRows()));
    nnet3::IoSpecification output_spec;
    output_spec.name = "output";
    output_spec.has_deriv = false;
    output_spec.indexes.resize(1);
    request.outputs.resize(1);
    request.outputs[0].Swap(&output_spec);
    shared_ptr<const nnet3::NnetComputation> computation = compiler->Compile(request);
    nnet3::Nnet *nnet_to_update = nullptr;  // we're not doing any update.
    nnet3::NnetComputer computer(nnet3::NnetComputeOptions(), *computation,
                    nnet, nnet_to_update);
    CuMatrix<BaseFloat> input_feats_cu(features);
    computer.AcceptInput("input", &input_feats_cu);
    computer.Run();
    CuMatrix<BaseFloat> cu_output;
    computer.GetOutputDestructive("output", &cu_output);
    xvector->Resize(cu_output.NumCols());
    xvector->CopyFromVec(cu_output.Row(0));
}

#define MIN_SPK_FEATS 50

bool KaldiRecognizer::GetSpkVectorVad(Vector<BaseFloat> &out_xvector, int *num_spk_frames)
{
	//spk_feature_->
	int num_frames = spk_feature_->NumFramesReady();
	std::cout << "num frames: " << num_frames << ' ' << frame_offset_ << ' ';
	Matrix<BaseFloat> mfcc(num_frames, spk_feature_->Dim());
	
	// Not very efficient, would be nice to have faster search
	int num_nonsilence_frames = 0;
	Vector<BaseFloat> feat(spk_feature_->Dim());
	Vector<BaseFloat> vad_result(num_frames);
	VadEnergyOptions optsVad;
	optsVad.vad_frames_context = 2;
	optsVad.vad_proportion_threshold = 0.12;
	optsVad.vad_energy_threshold = 5.5;
	optsVad.vad_energy_mean_scale = 0.5;

	for (int i = 0; i < num_frames; ++i) {
		spk_feature_->GetFrame(i, &feat);
		mfcc.CopyRowFromVec(feat, i);
	}
	//std::cout << "mfcc before: " << mfcc(1, 1) << ' ' << mfcc(2, 2) << '\n';
	ComputeVadEnergy(optsVad, mfcc, &vad_result);

	//apply cmvn
	SlidingWindowCmnOptions cmvn_opts;
	cmvn_opts.center = true;
	cmvn_opts.cmn_window = 300;
	Matrix<BaseFloat> features(mfcc.NumRows(), mfcc.NumCols(), kUndefined);
	SlidingWindowCmn(cmvn_opts, mfcc, &features);

	for (int i = 0; i < num_frames; ++i) {
		if (vad_result(i) < 0.01) {
			continue;
		}
		//mfcc.CopyRowFromVec(mfcc.Row(i), num_nonsilence_frames);
		features.CopyRowFromVec(features.Row(i), num_nonsilence_frames);
		num_nonsilence_frames++;
	}
	// Don't extract vector if not enough data
	std::cout << "num speaker frames: " <<num_nonsilence_frames << '\n';

	if (num_nonsilence_frames < MIN_SPK_FEATS) {
		return false;
	}

	//mfcc.Resize(num_nonsilence_frames, spk_feature_->Dim(), kCopyData);
	features.Resize(num_nonsilence_frames, spk_feature_->Dim(), kCopyData);

	//std::cout << "mfcc after resize: " << mfcc(1, 1) << ' ' << mfcc(2, 2) << '\n';
	
	
	/*for (int i = 0; i < num_frames; ++i) {
		if (vad_result(i) < 0.01) {
			continue;
		}
		//spk_feature_->GetFrame(i , &feat);
		features.CopyRowFromVec(features.Row(i), num_nonsilence_frames);
		//mfcc.CopyRowFromVec(feat, num_nonsilence_frames);
		num_nonsilence_frames++;
	}*/

	*num_spk_frames = num_nonsilence_frames;

	//features.Resize(num_nonsilence_frames, spk_feature_->Dim(), kCopyData);

	

	//nnet3::NnetSimpleComputationOptions opts;
	//nnet3::CachingOptimizingCompilerOptions compiler_config;
	//nnet3::CachingOptimizingCompiler compiler(spk_model_->speaker_nnet, opts.optimize_config, compiler_config);
	//std::cout << "features: " << features(1,1) << ' ' << features(2,2) << '\n';
	Vector<BaseFloat> xvector;
	//RunNnetComputation(features, spk_model_->speaker_nnet, &compiler, &xvector);
	RunNnetComputation(features, spk_model_->speaker_nnet, spk_model_->compiler, &xvector);

	// Whiten the vector with global mean and transform and normalize mean
	xvector.AddVec(-1.0, spk_model_->mean);
	std::cout << "xvector: " << xvector(1) << ' ' << xvector(2) << '\n';
	
	BaseFloat norm = 1001;
	//sometimes calculations are incorrect
	//i dont know what is the cause
	//after AddMatVec xvector values are incorrectly too high
	int counter = 0;
	while ((norm > 1000)&&(counter<5))
	{	
		out_xvector.SetZero();

		out_xvector.Resize(spk_model_->transform.NumRows(), kSetZero);
		//check if some values are not actually zeros
		for (int i = 0; i < out_xvector.Dim(); i++)
		{
			if (out_xvector(i) != 0)
				std::cout << out_xvector(i) << " ";
		}
		std::cout << '\n';

		out_xvector.AddMatVec(1.0, spk_model_->transform, kNoTrans, xvector, 0.0);
		std::cout << "out_xvector: " << out_xvector(1) << ' ' << out_xvector(2) << '\n';
		if (abs(out_xvector(1)) < 0.00000001)
		{
			std::cout << "xvector ERRROR AddMatVec: out_xvector1 == 0" << '\n';
		}
		

		norm = out_xvector.Norm(2.0);
		counter++;
		if (counter > 3)//Dont know how much should be
		{
			/*for (int i = 0; i < out_xvector.Dim(); i++)
			
			{
				std::cout << out_xvector(i)<<" ";				
			}
			std::cout << '\n';*/
			
			nnet3::NnetSimpleComputationOptions opts;
			nnet3::CachingOptimizingCompilerOptions compiler_config;
			nnet3::CachingOptimizingCompiler compiler(spk_model_->speaker_nnet, opts.optimize_config, compiler_config);
			//std::cout << "features: " << features(1,1) << ' ' << features(2,2) << '\n';
			Vector<BaseFloat> xvector1;
			RunNnetComputation(features, spk_model_->speaker_nnet, &compiler, &xvector1);
			//RunNnetComputation(features, spk_model_->speaker_nnet, spk_model_->compiler, &xvector);

			// Whiten the vector with global mean and transform and normalize mean
			xvector1.AddVec(-1.0, spk_model_->mean);

			for (int i = 0; i < xvector.Dim(); i++)
			{
				if (xvector(i)!= xvector1(i))
					std::cout << xvector(i)<<" " << xvector1(i) << "\n";
			}
			std::cout << '\n';
			
			out_xvector.Resize(spk_model_->transform.NumRows(), kSetZero);

			//check if some values are not actually zeros
			for (int i = 0; i < out_xvector.Dim(); i++)
			{
				if (out_xvector(i) != 0)
					std::cout << out_xvector(i) << " ";
			}
			std::cout << '\n';

			out_xvector.AddMatVec(1.0, spk_model_->transform, kNoTrans, xvector1, 0.0);
			norm = out_xvector.Norm(2.0);
		}
	}
	BaseFloat ratio = norm / sqrt(out_xvector.Dim()); // how much larger it is
												  // than it would be, in
												  // expectation, if normally
	std::cout << norm << '\n';
	if (norm == INFINITY)
	{
		std::cout << "xvector ERRROR: Norm INF" << '\n';
	}
	out_xvector.Scale(1.0 / ratio);
	std::cout << "xvector: " << out_xvector(1) << ' '<< out_xvector(2) << '\n';
	if (abs(out_xvector(1)) < 0.00000001)
	{
		std::cout << "xvector ERRROR: xvector1 == 0" << '\n';
	}
	return true;
}

bool KaldiRecognizer::GetSpkVectorVadMic(Vector<BaseFloat> &out_xvector, int *num_spk_frames, float rec_len)
{
	int32 chunk_length = int32(0.18*sample_frequency_);
	int32 total_chunks = 0;
	//get audio from mic until we end with pause
	std::cout << "Start Speaking\n";
	while (total_chunks<rec_len* sample_frequency_) {
		// Prepare the input audio samples
		Vector<BaseFloat> wave_part(chunk_length);
		bool ans = au_src_->Read(&wave_part);
		spk_feature_->AcceptWaveform(sample_frequency_, wave_part);
		total_chunks += chunk_length;
	}
	spk_feature_->InputFinished();
	//if (total_chunks) { //the end

	//}

	//get xvector
	int num_spk_frames1;
	if (GetSpkVectorVad(out_xvector, &num_spk_frames1)) {
		*num_spk_frames = num_spk_frames1;
		return true;
	}
	else return false;

}

bool KaldiRecognizer::GetSpkVector(Vector<BaseFloat> &out_xvector, int *num_spk_frames)
{
    vector<int32> nonsilence_frames;
    if (silence_weighting_->Active() && feature_pipeline_->NumFramesReady() > 0) {
        silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder(), true);
        silence_weighting_->GetNonsilenceFrames(feature_pipeline_->NumFramesReady(),
                                          frame_offset_ * 3,
                                          &nonsilence_frames);
    }

    int num_frames = spk_feature_->NumFramesReady() - frame_offset_ * 3;
    Matrix<BaseFloat> mfcc(num_frames, spk_feature_->Dim());

    // Not very efficient, would be nice to have faster search
    int num_nonsilence_frames = 0;
    Vector<BaseFloat> feat(spk_feature_->Dim());

    for (int i = 0; i < num_frames; ++i) {
       if (std::find(nonsilence_frames.begin(),
                     nonsilence_frames.end(), i / 3) == nonsilence_frames.end()) {
           continue;
       }

       spk_feature_->GetFrame(i + frame_offset_ * 3, &feat);
       mfcc.CopyRowFromVec(feat, num_nonsilence_frames);
       num_nonsilence_frames++;
    }

    *num_spk_frames = num_nonsilence_frames;

    // Don't extract vector if not enough data
    if (num_nonsilence_frames < MIN_SPK_FEATS) {
        return false;
    }

    mfcc.Resize(num_nonsilence_frames, spk_feature_->Dim(), kCopyData);

    SlidingWindowCmnOptions cmvn_opts;
    cmvn_opts.center = true;
    cmvn_opts.cmn_window = 300;
    Matrix<BaseFloat> features(mfcc.NumRows(), mfcc.NumCols(), kUndefined);
    SlidingWindowCmn(cmvn_opts, mfcc, &features);

    //nnet3::NnetSimpleComputationOptions opts;
    //nnet3::CachingOptimizingCompilerOptions compiler_config;
    //nnet3::CachingOptimizingCompiler compiler(spk_model_->speaker_nnet, opts.optimize_config, compiler_config);

    Vector<BaseFloat> xvector;
    RunNnetComputation(features, spk_model_->speaker_nnet, spk_model_->compiler, &xvector);

    // Whiten the vector with global mean and transform and normalize mean
    xvector.AddVec(-1.0, spk_model_->mean);

    out_xvector.Resize(spk_model_->transform.NumRows(), kSetZero);
    out_xvector.AddMatVec(1.0, spk_model_->transform, kNoTrans, xvector, 0.0);

    BaseFloat norm = out_xvector.Norm(2.0);
    BaseFloat ratio = norm / sqrt(out_xvector.Dim()); // how much larger it is
                                                  // than it would be, in
                                                  // expectation, if normally
    out_xvector.Scale(1.0 / ratio);

    return true;
}

const char *KaldiRecognizer::XvectorResult()
{
	spk_feature_->InputFinished();
	json::JSON obj;
	stringstream text;

	if (spk_model_) {
		Vector<BaseFloat> xvector;
		int num_spk_frames;
		if (GetSpkVectorVad(xvector, &num_spk_frames)) {
			for (int i = 0; i < xvector.Dim(); i++) {
				obj["spk"].append(xvector(i));
			}
			obj["spk_frames"] = num_spk_frames;
		}
	}

	return StoreReturn(obj.dump());
}

const char *KaldiRecognizer::GetSpksList(const char* path)
{
	KALDI_LOG << "\nGetting speaker list from: "<< path;
	std::string ark_path1(path);
	ark_path1.insert(0, "ark:");
	SequentialBaseFloatVectorReader test_ivector_reader(ark_path1);
	std::string out("");
	int32 num_test_ivectors = 0;
	int32 num_examples = 1;

	KALDI_LOG << "Reading test xVectors";
	for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
		std::string utt = test_ivector_reader.Key();
		if (num_test_ivectors == 0)
			out = utt;
		else
			out = out + string("\n")+ utt;
		KALDI_LOG << out<<"\n";
		num_test_ivectors++;
	}
	last_result_ = out;
	KALDI_LOG << "All speakers reading done \n";
	size_t len = out.length() + 1;
	delete out_str_;
	out_str_ = new char[len];
	strncpy_s(out_str_, len, out.c_str(), len);
	//out.c_str();
	return out_str_;
	//return last_result_.c_str();
}

bool KaldiRecognizer::DeleteSpeaker(const char* path, const char* user_id)
{
	std::string path1(path);

	if (path1.find(".wav") != std::string::npos)
		path1.replace(path1.length() - 4, 4, ".ark");
	else
		if (path1.find(".ark") == std::string::npos)
			path1.append(".ark");
	std::string path2(path1);
	path1.insert(0, "ark:");
	if (!std::experimental::filesystem::exists(path2))
	{
		return false;
	}
	else
	{
		//read previous ark and delete speaker xvector
		SequentialBaseFloatVectorReader test_ivector_reader(path1);

		KALDI_LOG << "Deleting speaker from ark file";
		BaseFloatVectorWriter vector_writer("ark:tmp.ark");
		for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {

			std::string utt = test_ivector_reader.Key();
			if (utt.compare(user_id) != 0)
			{
				vector_writer.Write(utt, test_ivector_reader.Value());
			}
		}
		
		vector_writer.Close();
		test_ivector_reader.Close();
		std::remove(path2.c_str());
		std::rename("tmp.ark", path2.c_str());

	}
	//vector_writer.Write(utt_id, xvector);
	return true;

}


const char *KaldiRecognizer::GetIdentityMic(const char* path, float rec_len, float& top_score)
{
	std::string out("");
	int32 dim = spk_model_->plda.Dim();
	int32 num_examples = 1;
	PldaConfig plda_config;
	plda_config.normalize_length = true;
	Vector<BaseFloat> xvector = this->GetXVectorMic(rec_len);
	float max_score = -1000;

	if (xvector.Dim() > 0)
	{
		Vector<BaseFloat> *transformed_ivector2 = new Vector<BaseFloat>(dim);
		spk_model_->plda.TransformIvector(plda_config, xvector, num_examples, transformed_ivector2);

		std::string ark_path1(path);
		ark_path1.insert(0, "ark:");
		SequentialBaseFloatVectorReader test_ivector_reader(ark_path1);
		


		
		float score;

		for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
			std::string utt = test_ivector_reader.Key();
			const Vector<BaseFloat> &ivector = test_ivector_reader.Value();
			Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

			spk_model_->plda.TransformIvector(plda_config, ivector, num_examples, transformed_ivector);

			Vector<double> train_ivector_dbl(*transformed_ivector), 
				test_ivector_dbl(*transformed_ivector2);
			score = spk_model_->plda.LogLikelihoodRatio(train_ivector_dbl, num_examples, test_ivector_dbl);
			if (score > max_score)
			{
				max_score = score;
				out = utt;
			}
		}
	}
	top_score = max_score;
	return out.c_str();
}
											

const char *KaldiRecognizer::MbrResult(CompactLattice &rlat)
{
    CompactLattice aligned_lat;
    if (model_->winfo_) {
        WordAlignLattice(rlat, *model_->trans_model_, *model_->winfo_, 0, &aligned_lat);
    } else {
        aligned_lat = rlat;
    }

    MinimumBayesRisk mbr(aligned_lat);
    const vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
    const vector<int32> &words = mbr.GetOneBest();
    const vector<pair<BaseFloat, BaseFloat> > &times =
          mbr.GetOneBestTimes();

    int size = words.size();

    json::JSON obj;
    stringstream text;

    // Create JSON object
    for (int i = 0; i < size; i++) {
        json::JSON word;

        if (words_) {
            word["word"] = model_->word_syms_->Find(words[i]);
            word["start"] = samples_round_start_ / sample_frequency_ + (frame_offset_ + times[i].first) * 0.03;
            word["end"] = samples_round_start_ / sample_frequency_ + (frame_offset_ + times[i].second) * 0.03;
            word["conf"] = conf[i];
            obj["result"].append(word);
        }

        if (i) {
            text << " ";
        }
        text << model_->word_syms_->Find(words[i]);
    }
    obj["text"] = text.str();

    if (spk_model_) {
        Vector<BaseFloat> xvector;
        int num_spk_frames;
        if (GetSpkVector(xvector, &num_spk_frames)) {
            for (int i = 0; i < xvector.Dim(); i++) {
                obj["spk"].append(xvector(i));
            }
            obj["spk_frames"] = num_spk_frames;
        }
    }

    return StoreReturn(obj.dump());
}

static bool CompactLatticeToWordAlignmentWeight(const CompactLattice &clat,
                                                std::vector<int32> *words,
                                                std::vector<int32> *begin_times,
                                                std::vector<int32> *lengths,
                                                CompactLattice::Weight *tot_weight_out)
{
  typedef CompactLattice::Arc Arc;
  typedef Arc::Label Label;
  typedef CompactLattice::StateId StateId;
  typedef CompactLattice::Weight Weight;
  using namespace fst;

  words->clear();
  begin_times->clear();
  lengths->clear();
  *tot_weight_out = Weight::Zero();

  StateId state = clat.Start();
  Weight tot_weight = Weight::One();

  int32 cur_time = 0;
  if (state == kNoStateId) {
    KALDI_WARN << "Empty lattice.";
    return false;
  }
  while (1) {
    Weight final = clat.Final(state);
    size_t num_arcs = clat.NumArcs(state);
    if (final != Weight::Zero()) {
      if (num_arcs != 0) {
        KALDI_WARN << "Lattice is not linear.";
        return false;
      }
      if (!final.String().empty()) {
        KALDI_WARN << "Lattice has alignments on final-weight: probably "
            "was not word-aligned (alignments will be approximate)";
      }
      tot_weight = Times(final, tot_weight);
      *tot_weight_out = tot_weight;
      return true;
    } else {
      if (num_arcs != 1) {
        KALDI_WARN << "Lattice is not linear: num-arcs = " << num_arcs;
        return false;
      }
      fst::ArcIterator<CompactLattice> aiter(clat, state);
      const Arc &arc = aiter.Value();
      Label word_id = arc.ilabel; // Note: ilabel==olabel, since acceptor.
      // Also note: word_id may be zero; we output it anyway.
      int32 length = arc.weight.String().size();
      words->push_back(word_id);
      begin_times->push_back(cur_time);
      lengths->push_back(length);
      tot_weight = Times(arc.weight, tot_weight);
      cur_time += length;
      state = arc.nextstate;
    }
  }
}


const char *KaldiRecognizer::NbestResult(CompactLattice &clat)
{
    Lattice lat;
    Lattice nbest_lat;
    std::vector<Lattice> nbest_lats;

    ConvertLattice (clat, &lat);
    fst::ShortestPath(lat, &nbest_lat, max_alternatives_);
    fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

    json::JSON obj;
    std::stringstream ss;
    for (int k = 0; k < nbest_lats.size(); k++) {

      Lattice nlat = nbest_lats[k];
      RmEpsilon(&nlat);
      CompactLattice nclat;
      CompactLattice aligned_nclat;
      ConvertLattice(nlat, &nclat);

      if (model_->winfo_) {
          WordAlignLattice(nclat, *model_->trans_model_, *model_->winfo_, 0, &aligned_nclat);
      } else {
          aligned_nclat = nclat;
      }

      std::vector<int32> words;
      std::vector<int32> begin_times;
      std::vector<int32> lengths;
      CompactLattice::Weight weight;

      CompactLatticeToWordAlignmentWeight(aligned_nclat, &words, &begin_times, &lengths, &weight);
      float likelihood = -(weight.Weight().Value1() + weight.Weight().Value2());

      stringstream text;
      json::JSON entry;

      for (int i = 0; i < words.size(); i++) {
        json::JSON word;
        if (words[i] == 0)
            continue;
        if (words_) {
            word["word"] = model_->word_syms_->Find(words[i]);
            word["start"] = samples_round_start_ / sample_frequency_ + (frame_offset_ + begin_times[i]) * 0.03;
            word["end"] = samples_round_start_ / sample_frequency_ + (frame_offset_ + begin_times[i] + lengths[i]) * 0.03;
            entry["result"].append(word);
        }
        if (i)
          text << " ";
        text << model_->word_syms_->Find(words[i]);
      }

      entry["text"] = text.str();
      entry["confidence"]= likelihood;
      obj["alternatives"].append(entry);
    }

    return StoreReturn(obj.dump());
}

const char* KaldiRecognizer::GetResult()
{
	if (model_) {			  
    if (decoder_->NumFramesDecoded() == 0) {
        return StoreEmptyReturn();
    }

    kaldi::CompactLattice clat;
    kaldi::CompactLattice rlat;
    decoder_->GetLattice(true, &clat);

    if (model_->rnnlm_lm_fst_) {
        kaldi::ComposeLatticePrunedOptions compose_opts;
        compose_opts.lattice_compose_beam = 3.0;
        compose_opts.max_arcs = 3000;

        TopSortCompactLatticeIfNeeded(&clat);
        fst::ComposeDeterministicOnDemandFst<fst::StdArc> combined_lms(lm_to_subtract_det_scale, lm_to_add);
        CompactLattice composed_clat;
        ComposeCompactLatticePruned(compose_opts, clat,
                                    &combined_lms, &rlat);
        lm_to_add_orig->Clear();
    } else if (model_->std_lm_fst_) {
        Lattice lat1;

        ConvertLattice(clat, &lat1);
        fst::ScaleLattice(fst::GraphLatticeScale(-1.0), &lat1);
        fst::ArcSort(&lat1, fst::OLabelCompare<kaldi::LatticeArc>());
        kaldi::Lattice composed_lat;
        fst::Compose(lat1, *lm_fst_, &composed_lat);
        fst::Invert(&composed_lat);
        kaldi::CompactLattice determinized_lat;
        DeterminizeLattice(composed_lat, &determinized_lat);
        fst::ScaleLattice(fst::GraphLatticeScale(-1), &determinized_lat);
        fst::ArcSort(&determinized_lat, fst::OLabelCompare<kaldi::CompactLatticeArc>());

        kaldi::ConstArpaLmDeterministicFst const_arpa_fst(model_->const_arpa_);
        kaldi::CompactLattice composed_clat;
        kaldi::ComposeCompactLatticeDeterministic(determinized_lat, &const_arpa_fst, &composed_clat);
        kaldi::Lattice composed_lat1;
        ConvertLattice(composed_clat, &composed_lat1);
        fst::Invert(&composed_lat1);
        DeterminizeLattice(composed_lat1, &rlat);
    } else {
        rlat = clat;
    }

    fst::ScaleLattice(fst::GraphLatticeScale(0.9), &rlat); // Apply rescoring weight

    if (max_alternatives_ == 0) {
        return MbrResult(rlat);
    } else {
        return NbestResult(rlat);
    }
	}
	else return XvectorResult();
}


const char* KaldiRecognizer::PartialResult()
{
    if (state_ != RECOGNIZER_RUNNING) {
        return StoreEmptyReturn();
    }

    json::JSON res;

    if (decoder_->NumFramesDecoded() == 0) {
        res["partial"] = "";
        return StoreReturn(res.dump());
    }

    kaldi::Lattice lat;
    decoder_->GetBestPath(false, &lat);
    vector<kaldi::int32> alignment, words;
    LatticeWeight weight;
    GetLinearSymbolSequence(lat, &alignment, &words, &weight);

    ostringstream text;
    for (size_t i = 0; i < words.size(); i++) {
        if (i) {
            text << " ";
        }
        text << model_->word_syms_->Find(words[i]);
    }
    res["partial"] = text.str();

    return StoreReturn(res.dump());
}

const char* KaldiRecognizer::Result()
{
    if (state_ != RECOGNIZER_RUNNING) {
        return StoreEmptyReturn();
    }
    decoder_->FinalizeDecoding();
    state_ = RECOGNIZER_ENDPOINT;
    return GetResult();
}

const char* KaldiRecognizer::FinalResult()
{
    if (state_ != RECOGNIZER_RUNNING) {
        return StoreEmptyReturn();
    }
	if (model_) {
    feature_pipeline_->InputFinished();
    UpdateSilenceWeights();
    decoder_->AdvanceDecoding();
    decoder_->FinalizeDecoding();
	}
    state_ = RECOGNIZER_FINALIZED;
    GetResult();

    // Free some memory while we are finalized, next
    // iteration will reinitialize them anyway
    delete decoder_;
    delete feature_pipeline_;
    delete silence_weighting_;
    delete spk_feature_;

    feature_pipeline_ = nullptr;
    silence_weighting_ = nullptr;
    decoder_ = nullptr;
    spk_feature_ = nullptr;

    return last_result_.c_str();
}

Vector<BaseFloat> KaldiRecognizer::GetXVector()
{
	Vector<BaseFloat> xvector;
	if (state_ != RECOGNIZER_RUNNING) {
		return xvector;
	}
	if (model_) {
		feature_pipeline_->InputFinished();
		UpdateSilenceWeights();
		decoder_->AdvanceDecoding();
		decoder_->FinalizeDecoding();
	}
	state_ = RECOGNIZER_FINALIZED;
	
	int num_spk_frames;
	if (model_) {
		spk_feature_->InputFinished();
		GetSpkVector(xvector, &num_spk_frames);
	}
	else
	{
		spk_feature_->InputFinished();
		GetSpkVectorVad(xvector, &num_spk_frames);
	}

	//CleanUp();

	return xvector;
}

Vector<BaseFloat> KaldiRecognizer::GetXVectorMic(float rec_len)
{
	Vector<BaseFloat> xvector;
	if (!(state_ == RECOGNIZER_RUNNING || state_ == RECOGNIZER_INITIALIZED)) {
		CleanUp();
	}
	state_ = RECOGNIZER_RUNNING;
	if (model_) {
		feature_pipeline_->InputFinished();
		UpdateSilenceWeights();
		decoder_->AdvanceDecoding();
		decoder_->FinalizeDecoding();
	}
	

	int num_spk_frames;
	if (model_)
		GetSpkVector(xvector, &num_spk_frames);//ToDo
	else
		GetSpkVectorVadMic(xvector, &num_spk_frames, rec_len);
	state_ = RECOGNIZER_FINALIZED;
	// Free some memory while we are finalized, next
	// iteration will reinitialize them anyway

	delete decoder_;
	delete feature_pipeline_;
	delete silence_weighting_;
	delete spk_feature_;

	feature_pipeline_ = nullptr;
	silence_weighting_ = nullptr;
	decoder_ = nullptr;
	spk_feature_ = nullptr;

	return xvector;
}

BaseFloat KaldiRecognizer::Plda2Score(Vector<BaseFloat> train, Vector<BaseFloat> test)
{
	int32 dim = spk_model_->plda.Dim();
	PldaConfig plda_config;
	plda_config.normalize_length = true;
	Vector<BaseFloat> *transformed_train = new Vector<BaseFloat>(dim);
	Vector<BaseFloat> *transformed_test = new Vector<BaseFloat>(dim);
	int32 num_examples = 1;
	spk_model_->plda.TransformIvector(plda_config, train,  num_examples, transformed_train);
	spk_model_->plda.TransformIvector(plda_config, test, num_examples, transformed_test);
	Vector<double> train_ivector_dbl(*transformed_train), test_ivector_dbl(*transformed_test);
	int32 num_train_examples = 1;
	BaseFloat score = spk_model_->plda.LogLikelihoodRatio(train_ivector_dbl, num_train_examples, test_ivector_dbl);
	return score;
}

bool KaldiRecognizer::PldaTrials(const char *ark_path, const char *trials_path, const char *out_path)
{
	int32 dim = spk_model_->plda.Dim();
	PldaConfig plda_config;
	plda_config.normalize_length = true;

	std::string ark_path1(ark_path);
	ark_path1.insert(0, "ark:");
	SequentialBaseFloatVectorReader test_ivector_reader(ark_path1);

	typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

	// These hashes will contain the iVectors in the PLDA subspace
	// (that makes the within-class variance unit and diagonalizes the
	// between-class covariance).  They will also possibly be length-normalized,
	// depending on the config.
	HashType test_ivectors;
	int32 num_test_ivectors = 0;
	int32 num_examples = 1;

	KALDI_LOG << "Reading test xVectors";
	for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
		std::string utt = test_ivector_reader.Key();
		if (test_ivectors.count(utt) != 0) {
			KALDI_ERR << "Duplicate test xVector found for utterance " << utt;
		}
		const Vector<BaseFloat> &ivector = test_ivector_reader.Value();
		Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

		spk_model_->plda.TransformIvector(plda_config, ivector,
			num_examples,
			transformed_ivector);
		test_ivectors[utt] = transformed_ivector;
		num_test_ivectors++;
	}
	KALDI_LOG << "Read " << num_test_ivectors << " test iVectors.";
	if (num_test_ivectors == 0)
		KALDI_ERR << "No test xVectors present.";

	Input ki(trials_path);
	bool binary = false;
	Output ko(out_path, binary);

	double sum = 0.0, sumsq = 0.0;
	std::string line;

	while (std::getline(ki.Stream(), line)) {
		std::vector<std::string> fields;
		SplitStringToVector(line, " \t\n\r", true, &fields);
		if (fields.size() != 3) {
			KALDI_ERR << "Bad line " 
				<< "in input (expected two fields: label key1 key2): " << line;
		}
		std::string label = fields[0], key1 = fields[1], key2 = fields[2];

		if (test_ivectors.count(key2) == 0) {
			KALDI_WARN << "Key " << key2 << " not present in test iVectors.";

			continue;
		}
		const Vector<BaseFloat> *train_ivector = test_ivectors[key1],
			*test_ivector = test_ivectors[key2];

		Vector<double> train_ivector_dbl(*train_ivector),
			test_ivector_dbl(*test_ivector);

		int32 num_train_examples = 1;

		BaseFloat score = spk_model_->plda.LogLikelihoodRatio(train_ivector_dbl,
			num_train_examples,
			test_ivector_dbl);


		ko.Stream() << key1 << ' ' << key2 << ' ' << score << ' '<< label << std::endl;
	}

	for (HashType::iterator iter = test_ivectors.begin();
		iter != test_ivectors.end(); ++iter)
		delete iter->second;


//	BaseFloat score = spk_model_->plda.LogLikelihoodRatio(train_ivector_dbl, num_train_examples, test_ivector_dbl);
	return true;
}

BaseFloat ComputeEer(std::vector<BaseFloat> *target_scores,
	std::vector<BaseFloat> *nontarget_scores,
	BaseFloat *threshold) {
	KALDI_ASSERT(!target_scores->empty() && !nontarget_scores->empty());
	std::sort(target_scores->begin(), target_scores->end());
	std::sort(nontarget_scores->begin(), nontarget_scores->end());

	size_t target_position = 0,
		target_size = target_scores->size();
	for (; target_position + 1 < target_size; target_position++) {
		ssize_t nontarget_size = nontarget_scores->size(),
			nontarget_n = nontarget_size * target_position * 1.0 / target_size,
			nontarget_position = nontarget_size - 1 - nontarget_n;
		if (nontarget_position < 0)
			nontarget_position = 0;
		if ((*nontarget_scores)[nontarget_position] <
			(*target_scores)[target_position])
			break;
	}
	*threshold = (*target_scores)[target_position];
	BaseFloat eer = target_position * 1.0 / target_size;
	return eer;
}

bool KaldiRecognizer::GetEer(const char *scores_rxfilename)
{
	std::vector<BaseFloat> target_scores, nontarget_scores;
	Input ki(scores_rxfilename);

	std::string line;
	while (std::getline(ki.Stream(), line)) {
		std::vector<std::string> split_line;
		SplitStringToVector(line, " \t", true, &split_line);
		BaseFloat score;
		if (split_line.size() != 4) {
			KALDI_ERR << "Invalid input line (must have 4 fields): "
				<< line;
			return false;
		}
		if (!ConvertStringToReal(split_line[2], &score)) {
			KALDI_ERR << "Invalid input line (first field must be float): "
				<< line;
			return false;
		}
		
		if (split_line[3][0] == '1')
			target_scores.push_back(score);
		else if (split_line[3][0] == '0')
			nontarget_scores.push_back(score);
		else {
			KALDI_ERR << "Invalid input line (second field must be "
				<< "'1' or '0')";
			return false;
		}
	}
	if (target_scores.empty() && nontarget_scores.empty())
		KALDI_ERR << "Empty input.";
	if (target_scores.empty())
		KALDI_ERR << "No target scores seen.";
	if (nontarget_scores.empty())
		KALDI_ERR << "No non-target scores seen.";

	BaseFloat threshold;
	BaseFloat eer = ComputeEer(&target_scores, &nontarget_scores, &threshold);

	KALDI_LOG << "Equal error rate is " << (100.0 * eer)
		<< "%, at threshold " << threshold;

	std::cout.precision(4);
	std::cout << (100.0 * eer);

	return true;

}

BaseFloat KaldiRecognizer::Cos2Score(Vector<BaseFloat> train, Vector<BaseFloat> test,bool norm)
{
	//int32 dim = spk_model_->plda.Dim();
	BaseFloat score = VecVec(train, test);
	if (norm)
		score /= train.Norm(2)*test.Norm(2);	
	return score;
}

void KaldiRecognizer::Reset()
{
    if (state_ == RECOGNIZER_RUNNING) {
        decoder_->FinalizeDecoding();
    }
    StoreEmptyReturn();
    state_ = RECOGNIZER_ENDPOINT;
}

const char *KaldiRecognizer::StoreEmptyReturn()
{
    if (!max_alternatives_) {
        return StoreReturn("{\"text\": \"\"}");
    } else {
        return StoreReturn("{\"alternatives\" : [{\"text\": \"\", \"confidence\" : 1.0}] }");
    }
}

// Store result in recognizer and return as const string
const char *KaldiRecognizer::StoreReturn(const string &res)
{
    last_result_ = res;
    return last_result_.c_str();
}
