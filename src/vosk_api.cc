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
#include "kaldi_recognizer.h"
#include "model.h"
#include "spk_model.h"

#if HAVE_CUDA
#include "cudamatrix/cu-device.h"
#endif

#include <string.h>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;

using namespace kaldi;

VoskModel *vosk_model_new(const char *model_path)
{
    return (VoskModel *)new Model(model_path);
}

void vosk_model_free(VoskModel *model)
{
    ((Model *)model)->Unref();
}

int vosk_model_find_word(VoskModel *model, const char *word)
{
    return (int) ((Model *)model)->FindWord(word);
}

VoskSpkModel *vosk_spk_model_new(const char *model_path)
{
    return (VoskSpkModel *)new SpkModel(model_path);
}

void vosk_spk_model_free(VoskSpkModel *model)
{
    ((SpkModel *)model)->Unref();
}

VoskRecognizer *vosk_recognizer_new(VoskModel *model, float sample_rate)
{
    return (VoskRecognizer *)new KaldiRecognizer((Model *)model, sample_rate);
}

VoskRecognizer *vosk_recognizer_new_spk(VoskModel *model, float sample_rate, VoskSpkModel *spk_model)
{
    return (VoskRecognizer *)new KaldiRecognizer((Model *)model, sample_rate, (SpkModel *)spk_model);
}

VoskRecognizer *vosk_recognizer_new_spk_no_model(VoskSpkModel *spk_model, bool need_mic)
{
	return (VoskRecognizer *)new KaldiRecognizer((SpkModel *)spk_model, need_mic);
}

VoskRecognizer *vosk_recognizer_new_grm(VoskModel *model, float sample_rate, const char *grammar)
{
    return (VoskRecognizer *)new KaldiRecognizer((Model *)model, sample_rate, grammar);
}

void vosk_recognizer_set_max_alternatives(VoskRecognizer *recognizer, int max_alternatives)
{
    ((KaldiRecognizer *)recognizer)->SetMaxAlternatives(max_alternatives);
}

void vosk_recognizer_set_words(VoskRecognizer *recognizer, int words)
{
    ((KaldiRecognizer *)recognizer)->SetWords((bool)words);
}

void vosk_recognizer_set_spk_model(VoskRecognizer *recognizer, VoskSpkModel *spk_model)
{
    ((KaldiRecognizer *)recognizer)->SetSpkModel((SpkModel *)spk_model);
}

int vosk_recognizer_accept_waveform(VoskRecognizer *recognizer, const char *data, int length)
{
    return ((KaldiRecognizer *)(recognizer))->AcceptWaveform(data, length);
}

int vosk_recognizer_accept_waveform_s(VoskRecognizer *recognizer, const short *data, int length)
{
    return ((KaldiRecognizer *)(recognizer))->AcceptWaveform(data, length);
}

int vosk_recognizer_accept_waveform_f(VoskRecognizer *recognizer, const float *data, int length)
{
    return ((KaldiRecognizer *)(recognizer))->AcceptWaveform(data, length);
}

const char *vosk_recognizer_result(VoskRecognizer *recognizer)
{
    return ((KaldiRecognizer *)recognizer)->Result();
}

const char *vosk_recognizer_partial_result(VoskRecognizer *recognizer)
{
    return ((KaldiRecognizer *)recognizer)->PartialResult();
}

const char *vosk_recognizer_final_result(VoskRecognizer *recognizer)
{
    return ((KaldiRecognizer *)recognizer)->FinalResult();
}

/*Vector<BaseFloat> vosk_getXVector(VoskRecognizer *recognizer)
{
	return ((KaldiRecognizer *)recognizer)->GetXVector();
}

BaseFloat vosk_plda2Score(VoskRecognizer * recognizer, Vector<BaseFloat> train, Vector<BaseFloat> test)
{
	return ((KaldiRecognizer *)recognizer)->Plda2Score(train,test);
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
	//((KaldiRecognizer*)recognizer).spk_feature_->InputFinished();
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
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVectorMic(rec_len);
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
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVector();
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
	Vector<BaseFloat> trainxv = ((KaldiRecognizer *)recognizer)->GetXVector();
	read_wav(recognizer, datatest);	
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVector();

	return ((KaldiRecognizer *)recognizer)->Plda2Score(trainxv, testxv);
}

float vosk_cos2Score(VoskRecognizer *recognizer, const char *datatrain, const char *datatest, bool norm)
{
	read_wav(recognizer, datatrain);
	Vector<BaseFloat> trainxv = ((KaldiRecognizer *)recognizer)->GetXVector();
	
	read_wav(recognizer, datatest);
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVector();

	//save_xvector(datatrain, trainxv, "test11");
	//save_xvector(datatest, testxv, "test22");


	return ((KaldiRecognizer *)recognizer)->Cos2Score(trainxv, testxv, norm);
}

float vosk_cosScoreMic(VoskRecognizer *recognizer, const char *datatrain, bool norm)
{
	read_wav(recognizer, datatrain);
	Vector<BaseFloat> trainxv = ((KaldiRecognizer *)recognizer)->GetXVector();
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVectorMic(4);
	if (testxv.Dim() != 0)
		return ((KaldiRecognizer *)recognizer)->Cos2Score(trainxv, testxv, norm);
	else 
		return -1000;
}

float vosk_pldaScoreMic(VoskRecognizer *recognizer, const char *datatrain)
{
	read_wav(recognizer, datatrain);
	Vector<BaseFloat> trainxv = ((KaldiRecognizer *)recognizer)->GetXVector();
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVectorMic(4);
	if (testxv.Dim() != 0)
		return ((KaldiRecognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

float vosk_plda_score_mic(VoskRecognizer *recognizer, const char *spk_path, const char *spk_id)
{
	Vector<BaseFloat> trainxv = load_xvector(spk_path, spk_id);
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVectorMic(4);
	if (testxv.Dim() != 0 && trainxv.Dim() != 0)
		return ((KaldiRecognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

float vosk_plda_score_mic_len(VoskRecognizer *recognizer, const char *spk_path, const char *spk_id, float rec_len)
{
	Vector<BaseFloat> trainxv = load_xvector(spk_path, spk_id);
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVectorMic(rec_len);
	if (testxv.Dim() != 0 && trainxv.Dim() != 0)
		return ((KaldiRecognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

float vosk_plda_score_wav(VoskRecognizer *recognizer, const char *spk_path, const char *spk_id, const char *wav_input_path)
{
	read_wav(recognizer, wav_input_path);
	Vector<BaseFloat> trainxv = load_xvector(spk_path, spk_id);
	Vector<BaseFloat> testxv = ((KaldiRecognizer *)recognizer)->GetXVector();
	
	if (testxv.Dim() != 0 && trainxv.Dim() != 0)
		return ((KaldiRecognizer *)recognizer)->Plda2Score(trainxv, testxv);
	else
		return -1000;
}

bool vosk_plda_score_trial(VoskRecognizer *recognizer, const char *ark_path, const char *trials_path, const char *out_path)
{
	if (((KaldiRecognizer *)recognizer)->PldaTrials(ark_path, trials_path, out_path))
	{
		((KaldiRecognizer *)recognizer)->GetEer(out_path);
		return true;
	}
	else
		return false;

}

bool vosk_get_eer(VoskRecognizer *recognizer, const char *scores_path)
{
	if (((KaldiRecognizer *)recognizer)->GetEer(scores_path))
		return true;
	else
		return false;

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
			Vector<BaseFloat> xvector = ((KaldiRecognizer *)recognizer)->GetXVector();

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

const char *vosk_get_speakers_list(VoskRecognizer *recognizer, const char* path)
{
	return ((KaldiRecognizer *)recognizer)->GetSpksList(path);
}
bool vosk_delete_user(VoskRecognizer *recognizer, const char* model_path, const char* user_id)
{
	return ((KaldiRecognizer *)recognizer)->DeleteSpeaker(model_path, user_id);
}

const char *vosk_get_ident_result(VoskRecognizer *recognizer, const char* path, float rec_len, float& top_score)
{
	return ((KaldiRecognizer *)recognizer)->GetIdentityMic(path, rec_len, top_score);
}


void vosk_recognizer_reset(VoskRecognizer *recognizer)
{
    ((KaldiRecognizer *)recognizer)->Reset();
}

void vosk_recognizer_free(VoskRecognizer *recognizer)
{
    delete (KaldiRecognizer *)(recognizer);
}

void vosk_set_log_level(int log_level)
{
    SetVerboseLevel(log_level);
}

void vosk_gpu_init()
{
#if HAVE_CUDA
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
