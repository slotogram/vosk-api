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

#include "vosk_api.h"

#include "recognizer.h"
#include "model.h"
#include "spk_model.h"

#if HAVE_CUDA
#include "cudamatrix/cu-device.h"
#include "batch_recognizer.h"
#endif

#include <string.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace kaldi;

VoskModel *vosk_model_new(const char *model_path)
{
    try {
        return (VoskModel *)new Model(model_path);
    } catch (...) {
        return nullptr;
    }
}

void vosk_model_free(VoskModel *model)
{
    if (model == nullptr) {
       return;
    }
    ((Model *)model)->Unref();
}

int vosk_model_find_word(VoskModel *model, const char *word)
{
    return (int) ((Model *)model)->FindWord(word);
}

VoskSpkModel *vosk_spk_model_new(const char *model_path)
{
    try {
        return (VoskSpkModel *)new SpkModel(model_path);
    } catch (...) {
        return nullptr;
    }
}

void vosk_spk_model_free(VoskSpkModel *model)
{
    if (model == nullptr) {
       return;
    }
    ((SpkModel *)model)->Unref();
}

VoskRecognizer *vosk_recognizer_new(VoskModel *model, float sample_rate)
{
    try {
        return (VoskRecognizer *)new Recognizer((Model *)model, sample_rate);
    } catch (...) {
        return nullptr;
    }
}

VoskRecognizer *vosk_recognizer_new_spk(VoskModel *model, float sample_rate, VoskSpkModel *spk_model)
{
    try {
        return (VoskRecognizer *)new Recognizer((Model *)model, sample_rate, (SpkModel *)spk_model);
    } catch (...) {
        return nullptr;
    }
}

VoskRecognizer *vosk_recognizer_new_spk_no_model(VoskSpkModel *spk_model, bool need_mic)
{
	return (VoskRecognizer *)new Recognizer((SpkModel *)spk_model, need_mic);
}

VoskRecognizer *vosk_recognizer_new_grm(VoskModel *model, float sample_rate, const char *grammar)
{
    try {
        return (VoskRecognizer *)new Recognizer((Model *)model, sample_rate, grammar);
    } catch (...) {
        return nullptr;
    }
}

void vosk_recognizer_set_max_alternatives(VoskRecognizer *recognizer, int max_alternatives)
{
    ((Recognizer *)recognizer)->SetMaxAlternatives(max_alternatives);
}

void vosk_recognizer_set_words(VoskRecognizer *recognizer, int words)
{
    ((Recognizer *)recognizer)->SetWords((bool)words);
}

void vosk_recognizer_set_partial_words(VoskRecognizer *recognizer, int partial_words)
{
    ((Recognizer *)recognizer)->SetPartialWords((bool)partial_words);
}

void vosk_recognizer_set_nlsml(VoskRecognizer *recognizer, int nlsml)
{
    ((Recognizer *)recognizer)->SetNLSML((bool)nlsml);
}

void vosk_recognizer_set_spk_model(VoskRecognizer *recognizer, VoskSpkModel *spk_model)
{
    if (recognizer == nullptr || spk_model == nullptr) {
       return;
    }
    ((Recognizer *)recognizer)->SetSpkModel((SpkModel *)spk_model);
}

int vosk_recognizer_accept_waveform(VoskRecognizer *recognizer, const char *data, int length)
{
    try {
        return ((Recognizer *)(recognizer))->AcceptWaveform(data, length);
    } catch (...) {
        return -1;
    }
}

int vosk_recognizer_accept_waveform_s(VoskRecognizer *recognizer, const short *data, int length)
{
    try {
        return ((Recognizer *)(recognizer))->AcceptWaveform(data, length);
    } catch (...) {
        return -1;
    }
}

int vosk_recognizer_accept_waveform_f(VoskRecognizer *recognizer, const float *data, int length)
{
    try {
        return ((Recognizer *)(recognizer))->AcceptWaveform(data, length);
    } catch (...) {
        return -1;
    }
}

const char *vosk_recognizer_result(VoskRecognizer *recognizer)
{
    return ((Recognizer *)recognizer)->Result();
}

const char *vosk_recognizer_partial_result(VoskRecognizer *recognizer)
{
    return ((Recognizer *)recognizer)->PartialResult();
}

const char *vosk_recognizer_final_result(VoskRecognizer *recognizer)
{
    return ((Recognizer *)recognizer)->FinalResult();
}

/*Vector<BaseFloat> vosk_getXVector(VoskRecognizer *recognizer)
{
	return ((Recognizer *)recognizer)->GetXVector();
}

BaseFloat vosk_plda2Score(VoskRecognizer * recognizer, Vector<BaseFloat> train, Vector<BaseFloat> test)
{
	return ((Recognizer *)recognizer)->Plda2Score(train,test);
}
*/

bool read_wav(VoskRecognizer *recognizer, const char* path)
{
	FILE *wavin;
	char buf[3200];
	int nread, final;

	fopen_s(&wavin, path, "rb");
	fseek(wavin, 44, SEEK_SET);
	while (!feof(wavin)) {
		nread = fread(buf, 1, sizeof(buf), wavin);
		final = vosk_recognizer_accept_waveform(recognizer, buf, nread);
	}
	fclose(wavin);
	//((Recognizer*)recognizer).spk_feature_->InputFinished();
	return true;
}

bool save_xvector(const char* path, Vector<BaseFloat> xvector, const char* utt_id)
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
		BaseFloatVectorWriter vector_writer(path1);
		vector_writer.Write(utt_id, xvector);
	}
	else
	{
		//read previous ark and add new speaker xvector
		SequentialBaseFloatVectorReader test_ivector_reader(path1);

		KALDI_LOG << "Saving speaker xVector in ark file";
		BaseFloatVectorWriter vector_writer("ark:tmp.ark");
		for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
			
			std::string utt = test_ivector_reader.Key();
			if (utt.compare(utt_id) != 0)
			{
				vector_writer.Write(utt, test_ivector_reader.Value());
			}
		}
		vector_writer.Write(utt_id, xvector);
		vector_writer.Close();
		test_ivector_reader.Close();
		std::remove(path2.c_str());
		std::rename("tmp.ark", path2.c_str());
		
	}
	//vector_writer.Write(utt_id, xvector);
	return true;
}

void combine_xv_arks(const char *temp_ark, const char *spk_ark)
{
	std::ofstream of_a(spk_ark, std::ios_base::binary | std::ios_base::app);
	std::ifstream if_b(temp_ark, std::ios_base::binary);

	of_a.seekp(0, std::ios_base::end);
	of_a << if_b.rdbuf();
	//of_a.flush();
}

Vector<BaseFloat> load_xvector(const char* path, const char* utt_id)
{
	Vector<BaseFloat> xvector;
	//load xvector from "spk.ark"
	std::string path1(path);
	path1.insert(0, "ark:");
	RandomAccessBaseFloatVectorReader ivector1_reader(path1);
	if (ivector1_reader.HasKey(utt_id))
		xvector = ivector1_reader.Value(utt_id);
	return xvector;
}

bool vosk_save_xvector_mic(VoskRecognizer *recognizer, const char *ark_path, const char *speaker_id, float rec_len)
{
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVectorMic(rec_len);
	if (testxv.Dim() != 0)
	{
		save_xvector(ark_path, testxv, speaker_id);
		return true;
	}
	return false;
}

bool vosk_save_xvector_wav(VoskRecognizer *recognizer, const char *ark_path, const char *speaker_id, const char *wav_path)
{
	read_wav(recognizer, wav_path);
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVector();
	if (testxv.Dim() != 0)
	{
		save_xvector(ark_path, testxv, speaker_id);
		return true;
	}
	return false;
}

float vosk_plda2Score(VoskRecognizer *recognizer, const char *datatrain, const char *datatest)
{
	read_wav(recognizer, datatrain);	
	Vector<BaseFloat> trainxv = ((Recognizer *)recognizer)->GetXVector();
	read_wav(recognizer, datatest);	
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVector();

	return ((Recognizer *)recognizer)->Plda2Score(trainxv, testxv);
}

float vosk_cos2Score(VoskRecognizer *recognizer, const char *datatrain, const char *datatest, bool norm)
{
	read_wav(recognizer, datatrain);
	Vector<BaseFloat> trainxv = ((Recognizer *)recognizer)->GetXVector();
	
	read_wav(recognizer, datatest);
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVector();

	//save_xvector(datatrain, trainxv, "test11");
	//save_xvector(datatest, testxv, "test22");


	return ((Recognizer *)recognizer)->Cos2Score(trainxv, testxv, norm);
}

float vosk_cosScoreMic(VoskRecognizer *recognizer, const char *datatrain, bool norm)
{
	read_wav(recognizer, datatrain);
	Vector<BaseFloat> trainxv = ((Recognizer *)recognizer)->GetXVector();
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVectorMic(4);
	if (testxv.Dim() != 0)
		return ((Recognizer *)recognizer)->Cos2Score(trainxv, testxv, norm);
	else 
		return -1000;
}

float vosk_pldaScoreMic(VoskRecognizer *recognizer, const char *datatrain)
{
	read_wav(recognizer, datatrain);
	Vector<BaseFloat> trainxv = ((Recognizer *)recognizer)->GetXVector();
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVectorMic(4);
	if (testxv.Dim() != 0)
		return ((Recognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

float vosk_plda_score_mic(VoskRecognizer *recognizer, const char *spk_path, const char *spk_id)
{
	Vector<BaseFloat> trainxv = load_xvector(spk_path, spk_id);
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVectorMic(4);
	if (testxv.Dim() != 0 && trainxv.Dim() != 0)
		return ((Recognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

float vosk_plda_score_mic_len(VoskRecognizer *recognizer, const char *spk_path, const char *spk_id, float rec_len)
{
	Vector<BaseFloat> trainxv = load_xvector(spk_path, spk_id);
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVectorMic(rec_len);
	if (testxv.Dim() != 0 && trainxv.Dim() != 0)
		return ((Recognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

float vosk_plda_score_wav(VoskRecognizer *recognizer, const char *spk_path, const char *spk_id, const char *wav_input_path)
{
	read_wav(recognizer, wav_input_path);
	Vector<BaseFloat> trainxv = load_xvector(spk_path, spk_id);
	Vector<BaseFloat> testxv = ((Recognizer *)recognizer)->GetXVector();
	
	if (testxv.Dim() != 0 && trainxv.Dim() != 0)
		return ((Recognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}
bool vosk_shuffle_trial_list(const char *trials_path, const char *out_path)
{
	Input ki(trials_path);
	bool binary = false;
	Output ko(out_path, binary);

	std::string line;
	bool get_first = true;
	while (std::getline(ki.Stream(), line)) {
		std::vector<std::string> fields;
		SplitStringToVector(line, " \t\n\r", true, &fields);
		if (get_first && fields.size() != 3) {
			KALDI_ERR << "Bad line "
				<< "in input (expected two fields: label key1 key2): " << line;
		}

		get_first = false;
		//std::string label = fields[0], key1 = fields[1], key2 = fields[2];
		//std::string label = fields[2], key1 = fields[1], key2 = fields[0];
		if (fields.size() == 3)
			ko.Stream() << fields[2] << ' ' << fields[1] << ' ' << fields[0] << std::endl;
	}
	ko.Close();
	return true;
}
bool vosk_plda_score_trial(VoskRecognizer *recognizer, const char *ark_path, const char *trials_path, const char *out_path)
{
	if (((Recognizer *)recognizer)->PldaTrials(ark_path, trials_path, out_path))
	{
		((Recognizer *)recognizer)->GetEer(out_path);
		return true;
	}
	else
		return false;

}

bool vosk_get_eer(VoskRecognizer *recognizer, const char *scores_path)
{
	if (((Recognizer *)recognizer)->GetEer(scores_path))
		return true;
	else
		return false;

}


bool vosk_create_speaker_xvectors(VoskRecognizer *recognizer, const char *ark_path, const char *ark_out_path)
{
	std::string ark_path1(ark_path), ark_out_path1(ark_out_path);
	ark_path1.insert(0, "ark:"); ark_out_path1.insert(0, "ark:");

	SequentialBaseFloatVectorReader test_ivector_reader(ark_path1);

	typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;
	typedef unordered_map<string, int32, StringHasher> HashType2;

	// These hashes will contain the iVectors in the PLDA subspace
	// (that makes the within-class variance unit and diagonalizes the
	// between-class covariance).  They will also possibly be length-normalized,
	// depending on the config.
	HashType test_ivectors;
	HashType avg_ivectors;
	HashType2 spk_count;
	int32 num_test_ivectors = 0;
	int32 num_examples = 1;
	int32 dim = 0;
	KALDI_LOG << "Reading test xVectors";
	for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
		std::string utt = test_ivector_reader.Key();
		if (test_ivectors.count(utt) != 0) {
			KALDI_ERR << "Duplicate test xVector found for utterance " << utt;
		}
		//get speaker_id
		
		std::string utt1 = utt;
		while (utt1.find('\\') != string::npos)
		{
			utt1 = utt1.substr(utt1.find('\\') + 1);
		}
		utt1 = utt1.substr(0,utt1.find('-'));
		

		const Vector<BaseFloat> &ivector = test_ivector_reader.Value();
		if (avg_ivectors.count(utt1) == 0) {
			avg_ivectors[utt1] = new Vector<BaseFloat>(ivector.Dim());
			*avg_ivectors[utt1] = ivector;
			spk_count[utt1] = 1;
			dim = ivector.Dim();
		}
		else
		{
			spk_count[utt1]++;
			avg_ivectors[utt1]->AddVec(1,ivector);// += ivector;
		}

		Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);
		*transformed_ivector = ivector;
		test_ivectors[utt] = transformed_ivector;
		num_test_ivectors++;
	}
	KALDI_LOG << "Read " << num_test_ivectors << " test iVectors.";
	if (num_test_ivectors == 0)
		KALDI_ERR << "No test xVectors present.";

	//get avg_spk_xvector
	Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);
	for (HashType::iterator iter = avg_ivectors.begin();
		iter != avg_ivectors.end(); ++iter)
	{
		
		transformed_ivector->Set((BaseFloat)spk_count[iter->first]);
		avg_ivectors[iter->first]->DivElements(*transformed_ivector);
		
	}
	delete transformed_ivector;
		//save avg_xvector

	BaseFloatVectorWriter vector_writer(ark_path1);
	for (HashType::iterator iter = avg_ivectors.begin();
		iter != avg_ivectors.end(); ++iter)
	{
		vector_writer.Write(iter->first, *iter->second);
	}
	vector_writer.Close();

	for (HashType::iterator iter = test_ivectors.begin();
		iter != test_ivectors.end(); ++iter)
		delete iter->second;
	for (HashType::iterator iter = avg_ivectors.begin();
		iter != avg_ivectors.end(); ++iter)
		delete iter->second;

	return true;
}

bool vosk_compute_voxceleb_xvectors(VoskRecognizer *recognizer, const char *ark_path, const char *voxceleb_path)
{
	std::string ark_path1(ark_path), vox_path(voxceleb_path);
	ark_path1.insert(0, "ark:");
	int offset = vox_path.length();
	std::string::size_type pos = 0;
	BaseFloatVectorWriter vector_writer(ark_path1);
	std::string wav_path;
	for (const auto & entry : fs::recursive_directory_iterator(voxceleb_path))
	{
		wav_path = entry.path().string();
		if (wav_path.find(std::string(".wav")) != std::string::npos)
		{
			read_wav(recognizer, wav_path.c_str());
			Vector<BaseFloat> xvector = ((Recognizer *)recognizer)->GetXVector();

			wav_path = wav_path.substr(offset + 1, wav_path.length() - offset - 1);
			//std::cout << wav_path << std::endl;
			pos = wav_path.find('\\');
			if (pos != std::string::npos)
			{
				wav_path[pos] = '/';
				pos = wav_path.find('\\');
				if (pos != std::string::npos)
					wav_path[pos] = '/';
			}
			vector_writer.Write(wav_path, xvector);
		}
	}
		//std::cout << entry.path() << std::endl; 
}

bool vosk_compute_path_xvectors(VoskRecognizer *recognizer, const char *ark_path, const char *voxceleb_path)
{
	std::string ark_path1(ark_path), vox_path(voxceleb_path);
	ark_path1.insert(0, "ark:");
	int offset = vox_path.length();
	std::string::size_type pos = 0;
	BaseFloatVectorWriter vector_writer(ark_path1);
	std::string wav_path;
	for (const auto & entry : fs::recursive_directory_iterator(voxceleb_path))
	{
		wav_path = entry.path().string();
		if (wav_path.find(std::string(".wav")) != std::string::npos)
		{
			read_wav(recognizer, wav_path.c_str());
			Vector<BaseFloat> xvector = ((Recognizer *)recognizer)->GetXVector();

			vector_writer.Write(wav_path, xvector);
		}
	}
	//std::cout << entry.path() << std::endl; 
}

const char *vosk_get_speakers_list(VoskRecognizer *recognizer, const char* path)
{
	return ((Recognizer *)recognizer)->GetSpksList(path);
}
bool vosk_delete_user(VoskRecognizer *recognizer, const char* model_path, const char* user_id)
{
	return ((Recognizer *)recognizer)->DeleteSpeaker(model_path, user_id);
}

const char *vosk_get_ident_result(VoskRecognizer *recognizer, const char* path, float rec_len, float& top_score)
{
	return ((Recognizer *)recognizer)->GetIdentityMic(path, rec_len, top_score);
}


void vosk_recognizer_reset(VoskRecognizer *recognizer)
{
    ((Recognizer *)recognizer)->Reset();
}

void vosk_recognizer_free(VoskRecognizer *recognizer)
{
    delete (Recognizer *)(recognizer);
}

void vosk_set_log_level(int log_level)
{
    SetVerboseLevel(log_level);
}

void vosk_gpu_init()
{
#if HAVE_CUDA
//    kaldi::CuDevice::EnableTensorCores(true);
//    kaldi::CuDevice::EnableTf32Compute(true);
    kaldi::CuDevice::Instantiate().SelectGpuId("yes");
    kaldi::CuDevice::Instantiate().AllowMultithreading();
#endif
}

void vosk_gpu_thread_init()
{
#if HAVE_CUDA
    kaldi::CuDevice::Instantiate();
#endif
}

VoskBatchModel *vosk_batch_model_new()
{
#if HAVE_CUDA
    return (VoskBatchModel *)(new BatchModel());
#else
    return NULL;
#endif
}

void vosk_batch_model_free(VoskBatchModel *model)
{
#if HAVE_CUDA
    delete ((BatchModel *)model);
#endif
}

void vosk_batch_model_wait(VoskBatchModel *model)
{
#if HAVE_CUDA
    ((BatchModel *)model)->WaitForCompletion();
#endif
}

VoskBatchRecognizer *vosk_batch_recognizer_new(VoskBatchModel *model, float sample_rate)
{
#if HAVE_CUDA
    return (VoskBatchRecognizer *)(new BatchRecognizer((BatchModel *)model, sample_rate));
#else
    return NULL;
#endif
}

void vosk_batch_recognizer_free(VoskBatchRecognizer *recognizer)
{
#if HAVE_CUDA
    delete ((BatchRecognizer *)recognizer);
#endif
}

void vosk_batch_recognizer_accept_waveform(VoskBatchRecognizer *recognizer, const char *data, int length)
{
#if HAVE_CUDA
    ((BatchRecognizer *)recognizer)->AcceptWaveform(data, length);
#endif
}

void vosk_batch_recognizer_set_nlsml(VoskBatchRecognizer *recognizer, int nlsml)
{
#if HAVE_CUDA
    ((BatchRecognizer *)recognizer)->SetNLSML((bool)nlsml);
#endif
}

void vosk_batch_recognizer_finish_stream(VoskBatchRecognizer *recognizer)
{
#if HAVE_CUDA
    ((BatchRecognizer *)recognizer)->FinishStream();
#endif
}

const char *vosk_batch_recognizer_front_result(VoskBatchRecognizer *recognizer)
{
#if HAVE_CUDA
    return ((BatchRecognizer *)recognizer)->FrontResult();
#else
    return NULL;
#endif
}

void vosk_batch_recognizer_pop(VoskBatchRecognizer *recognizer)
{
#if HAVE_CUDA
    ((BatchRecognizer *)recognizer)->Pop();
#endif
}


int vosk_batch_recognizer_get_pending_chunks(VoskBatchRecognizer *recognizer)
{
#if HAVE_CUDA
    return ((BatchRecognizer *)recognizer)->GetNumPendingChunks();
#else
    return 0;
#endif
}
