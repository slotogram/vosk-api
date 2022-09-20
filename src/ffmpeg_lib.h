// Copyright 2019-2021 Alpha Cephei Inc.
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

#ifndef VOSK_FFMPEG_LIB_H
#define VOSK_FFMPEG_LIB_H

#ifdef __cplusplus
extern "C" {
#endif

#include "libavformat/avformat.h"
#include "libavformat/avio.h"

#include "libavcodec/avcodec.h"

#include "libavutil/audio_fifo.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/frame.h"
#include "libavutil/opt.h"

#include "libswresample/swresample.h"

#ifdef __cplusplus
}
#endif

#include <stdio.h>

#ifndef __cplusplus
    typedef uint8_t bool;
    #define true 1
    #define false 0
#endif

#ifdef __cplusplus
    #define REINTERPRET_CAST(type, variable) reinterpret_cast<type>(variable)
    #define STATIC_CAST(type, variable) static_cast<type>(variable)
#else
    #define C_CAST(type, variable) ((type)variable)
    #define REINTERPRET_CAST(type, variable) C_CAST(type, variable)
    #define STATIC_CAST(type, variable) C_CAST(type, variable)
#endif

#ifdef av_err2str
#undef av_err2str
#include <string>
av_always_inline std::string av_err2string(int errnum) {
    char str[AV_ERROR_MAX_STRING_SIZE];
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#define av_err2str(err) av_err2string(err).c_str()
#endif  // av_err2str

#define RAW_OUT_ON_PLANAR true
/* The output bit rate in bit/s */
#define OUTPUT_BIT_RATE 96000
/* The number of output channels */
#define OUTPUT_CHANNELS 1

/**
 * Open an input file and the required decoder.
 * @param      filename             File to be opened
 * @param[out] input_format_context Format context of opened file
 * @param[out] input_codec_context  Codec context of opened file
 * @return Error code (0 if successful)
 */
int open_input_file(const char *filename,
                           AVFormatContext **input_format_context,
                           AVCodecContext **input_codec_context);

/**
 * Open an output file and the required encoder.
 * Also set some basic encoder parameters.
 * Some of these parameters are based on the input file's parameters.
 * @param      filename              File to be opened
 * @param      input_codec_context   Codec context of input file
 * @param[out] output_format_context Format context of output file
 * @param[out] output_codec_context  Codec context of output file
 * @return Error code (0 if successful)
 */
int open_output_file(const char *filename, const float sample_frequency,
                            AVCodecContext *input_codec_context,
                            AVFormatContext **output_format_context,
                            AVCodecContext **output_codec_context);

/**
 * Open an output context and the required encoder without saving to file.
 * Also set some basic encoder parameters.
 * @param      input_codec_context   Codec context of input file
 * @param[out] output_format_context Format context of output file
 * @param[out] output_codec_context  Codec context of output file
 * @return Error code (0 if successful)
 */
int open_output_context(const float sample_frequency,
                            AVCodecContext *input_codec_context,
                            AVCodecContext **output_codec_context);


/**
 * Initialize one data packet for reading or writing.
 * @param[out] packet Packet to be initialized
 * @return Error code (0 if successful)
 */
int init_packet(AVPacket **packet);


/**
 * Initialize one audio frame for reading from the input file.
 * @param[out] frame Frame to be initialized
 * @return Error code (0 if successful)
 */
int init_input_frame(AVFrame **frame);


/**
 * Initialize the audio resampler based on the input and output codec settings.
 * If the input and output sample formats differ, a conversion is required
 * libswresample takes care of this, but requires initialization.
 * @param      input_codec_context  Codec context of the input file
 * @param      output_codec_context Codec context of the output file
 * @param[out] resample_context     Resample context for the required conversion
 * @return Error code (0 if successful)
 */
int init_resampler(AVCodecContext *input_codec_context,
                          AVCodecContext *output_codec_context,
                          SwrContext **resample_context);

/**
 * Initialize a FIFO buffer for the audio samples to be encoded.
 * @param[out] fifo                 Sample buffer
 * @param      output_codec_context Codec context of the output file
 * @return Error code (0 if successful)
 */
int init_fifo(AVAudioFifo **fifo, AVCodecContext *output_codec_context);

/**
 * Write the header of the output file container.
 * @param output_format_context Format context of the output file
 * @return Error code (0 if successful)
 */
int write_output_file_header(AVFormatContext *output_format_context);


/**
 * Decode one audio frame from the input file.
 * @param      frame                Audio frame to be decoded
 * @param      input_format_context Format context of the input file
 * @param      input_codec_context  Codec context of the input file
 * @param[out] data_present         Indicates whether data has been decoded
 * @param[out] finished             Indicates whether the end of file has
 *                                  been reached and all data has been
 *                                  decoded. If this flag is false, there
 *                                  is more data to be decoded, i.e., this
 *                                  function has to be called again.
 * @return Error code (0 if successful)
 */
int decode_audio_frame(AVFrame *frame,
                              AVFormatContext *input_format_context,
                              AVCodecContext *input_codec_context,
                              int *data_present, int *finished);


/**
 * Initialize a temporary storage for the specified number of audio samples.
 * The conversion requires temporary storage due to the different format.
 * The number of audio samples to be allocated is specified in frame_size.
 * @param[out] converted_input_samples Array of converted samples. The
 *                                     dimensions are reference, channel
 *                                     (for multi-channel audio), sample.
 * @param      output_codec_context    Codec context of the output file
 * @param      frame_size              Number of samples to be converted in
 *                                     each round
 * @return Error code (0 if successful)
 */
int init_converted_samples(uint8_t ***converted_input_samples,
                                  AVCodecContext *output_codec_context,
                                  int frame_size);

/**
 * Convert the input audio samples into the output sample format.
 * The conversion happens on a per-frame basis, the size of which is
 * specified by frame_size.
 * @param      input_data       Samples to be decoded. The dimensions are
 *                              channel (for multi-channel audio), sample.
 * @param[out] converted_data   Converted samples. The dimensions are channel
 *                              (for multi-channel audio), sample.
 * @param      frame_size       Number of samples to be converted
 * @param      resample_context Resample context for the conversion
 * @return Error code (0 if successful)
 */
int convert_samples(const uint8_t **input_data,
                           uint8_t **converted_data, const int frame_size, const int out_frame_size, 
                           SwrContext *resample_context);

/**
 * Add converted input audio samples to the FIFO buffer for later processing.
 * @param fifo                    Buffer to add the samples to
 * @param converted_input_samples Samples to be added. The dimensions are channel
 *                                (for multi-channel audio), sample.
 * @param frame_size              Number of samples to be converted
 * @return Error code (0 if successful)
 */
int add_samples_to_fifo(AVAudioFifo *fifo,
                               uint8_t **converted_input_samples,
                               const int frame_size);

/**
 * Read one audio frame from the input file, decode, convert and store
 * it in the FIFO buffer.
 * @param      fifo                 Buffer used for temporary storage
 * @param      input_format_context Format context of the input file
 * @param      input_codec_context  Codec context of the input file
 * @param      output_codec_context Codec context of the output file
 * @param      resampler_context    Resample context for the conversion
 * @param[out] finished             Indicates whether the end of file has
 *                                  been reached and all data has been
 *                                  decoded. If this flag is false,
 *                                  there is more data to be decoded,
 *                                  i.e., this function has to be called
 *                                  again.
 * @return Error code (0 if successful)
 */
int read_decode_convert_and_store(AVAudioFifo *fifo,
                                         AVFormatContext *input_format_context,
                                         AVCodecContext *input_codec_context,
                                         AVCodecContext *output_codec_context,
                                         SwrContext *resampler_context,
                                         int *finished);


/**
 * Initialize one input frame for writing to the output file.
 * The frame will be exactly frame_size samples large.
 * @param[out] frame                Frame to be initialized
 * @param      output_codec_context Codec context of the output file
 * @param      frame_size           Size of the frame
 * @return Error code (0 if successful)
 */
int init_output_frame(AVFrame **frame,
                             AVCodecContext *output_codec_context,
                             int frame_size);

/* Global timestamp for the audio frames. */
static int64_t pts = 0;

/**
 * Encode one frame worth of audio to the output file.
 * @param      frame                 Samples to be encoded
 * @param      output_format_context Format context of the output file
 * @param      output_codec_context  Codec context of the output file
 * @param[out] data_present          Indicates whether data has been
 *                                   encoded
 * @return Error code (0 if successful)
 */
int encode_audio_frame(AVFrame *frame,
                              AVFormatContext *output_format_context,
                              AVCodecContext *output_codec_context,
                              int *data_present);

/**
 * Load one audio frame from the FIFO buffer, encode and write it to the
 * output file.
 * @param fifo                  Buffer used for temporary storage
 * @param output_format_context Format context of the output file
 * @param output_codec_context  Codec context of the output file
 * @return Error code (0 if successful)
 */
int load_encode_and_write(AVAudioFifo *fifo,
                                 AVFormatContext *output_format_context,
                                 AVCodecContext *output_codec_context);

/**
 * Write the trailer of the output file container.
 * @param output_format_context Format context of the output file
 * @return Error code (0 if successful)
 */
int write_output_file_trailer(AVFormatContext *output_format_context);


/**
 * Print an error string describing the errorCode to stderr.
 */
int printError(const char* prefix, int errorCode); 

/**
 * Extract a single sample and convert to float.
 */
float getSample(const AVCodecContext* codecCtx, uint8_t* buffer, int sampleIndex); 
/**
 * Write the frame to an output file.
 */
/*void handleFrame(const AVCodecContext* codecCtx, const AVFrame* frame) {
    if(av_sample_fmt_is_planar(codecCtx->sample_fmt) == 1) {
        // This means that the data of each channel is in its own buffer.
        // => frame->extended_data[i] contains data for the i-th channel.
        for(int s = 0; s < frame->nb_samples; ++s) {
            for(int c = 0; c < codecCtx->channels; ++c) {
                float sample = getSample(codecCtx, frame->extended_data[c], s);
                fwrite(&sample, sizeof(float), 1, outFile);
            }
        }
    } else {
        // This means that the data of each channel is in the same buffer.
        // => frame->extended_data[0] contains data of all channels.
        if(RAW_OUT_ON_PLANAR) {
            fwrite(frame->extended_data[0], 1, frame->linesize[0], outFile);
        } else {
            for(int s = 0; s < frame->nb_samples; ++s) {
                for(int c = 0; c < codecCtx->channels; ++c) {
                    float sample = getSample(codecCtx, frame->extended_data[0], s*codecCtx->channels+c);
                    fwrite(&sample, sizeof(float), 1, outFile);
                }
            }
        }
    }
}*/

/**
 * Find the first audio stream and returns its index. If there is no audio stream returns -1.
 */
/*int findAudioStream(const AVFormatContext* formatCtx) {
    int audioStreamIndex = -1;
    for(size_t i = 0; i < formatCtx->nb_streams; ++i) {
        // Use the first audio stream we can find.
        // NOTE: There may be more than one, depending on the file.
        if(formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            break;
        }
    }
    return audioStreamIndex;
}*/

/*
 * Print information about the input file and the used codec.
 */
void printStreamInformation(const AVCodec* codec, const AVCodecContext* codecCtx, int audioStreamIndex); 
/**
 * Receive as many frames as available and handle them.
 */
/*
int receiveAndHandle(AVCodecContext* codecCtx, AVFrame* frame) {
    int err = 0;
    // Read the packets from the decoder.
    // NOTE: Each packet may generate more than one frame, depending on the codec.
    while((err = avcodec_receive_frame(codecCtx, frame)) == 0) {
        // Let's handle the frame in a function.
        handleFrame(codecCtx, frame);
        // Free any buffers and reset the fields to default values.
        av_frame_unref(frame);
    }
    return err;
}

/*
 * Drain any buffered frames.
 */
/*
void drainDecoder(AVCodecContext* codecCtx, AVFrame* frame) {
    int err = 0;
    // Some codecs may buffer frames. Sending NULL activates drain-mode.
    if((err = avcodec_send_packet(codecCtx, NULL)) == 0) {
        // Read the remaining packets from the decoder.
        err = receiveAndHandle(codecCtx, frame);
        if(err != AVERROR(EAGAIN) && err != AVERROR_EOF) {
            // Neither EAGAIN nor EOF => Something went wrong.
            printError("Receive error.", err);
        }
    } else {
        // Something went wrong.
        printError("Send error.", err);
    }
}
*/

#endif /* VOSK_FFMPEG_LIB_H */
