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

#ifdef __cplusplus
extern "C" {
#endif

    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>

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

FILE* outFile;
// #define RAW_OUT_ON_PLANAR false
#define RAW_OUT_ON_PLANAR true

/**
 * Print an error string describing the errorCode to stderr.
 */
int printError(const char* prefix, int errorCode) {
    if(errorCode == 0) {
        return 0;
    } else {
        const size_t bufsize = 64;
        char buf[bufsize];

        if(av_strerror(errorCode, buf, bufsize) != 0) {
            strcpy(buf, "UNKNOWN_ERROR");
        }
        fprintf(stderr, "%s (%d: %s)\n", prefix, errorCode, buf);

        return errorCode;
    }
}

/**
 * Extract a single sample and convert to float.
 */
float getSample(const AVCodecContext* codecCtx, uint8_t* buffer, int sampleIndex) {
    int64_t val = 0;
    float ret = 0;
    int sampleSize = av_get_bytes_per_sample(codecCtx->sample_fmt);
    switch(sampleSize) {
        case 1:
            // 8bit samples are always unsigned
            val = REINTERPRET_CAST(uint8_t*, buffer)[sampleIndex];
            // make signed
            val -= 127;
            break;

        case 2:
            val = REINTERPRET_CAST(int16_t*, buffer)[sampleIndex];
            break;

        case 4:
            val = REINTERPRET_CAST(int32_t*, buffer)[sampleIndex];
            break;

        case 8:
            val = REINTERPRET_CAST(int64_t*, buffer)[sampleIndex];
            break;

        default:
            fprintf(stderr, "Invalid sample size %d.\n", sampleSize);
            return 0;
    }

    // Check which data type is in the sample.
    switch(codecCtx->sample_fmt) {
        case AV_SAMPLE_FMT_U8:
        case AV_SAMPLE_FMT_S16:
        case AV_SAMPLE_FMT_S32:
        case AV_SAMPLE_FMT_U8P:
        case AV_SAMPLE_FMT_S16P:
        case AV_SAMPLE_FMT_S32P:
            // integer => Scale to [-1, 1] and convert to float.
            ret = val / STATIC_CAST(float, ((1 << (sampleSize*8-1))-1));
            break;

        case AV_SAMPLE_FMT_FLT:
        case AV_SAMPLE_FMT_FLTP:
            // float => reinterpret
            ret = *REINTERPRET_CAST(float*, &val);
            break;

        case AV_SAMPLE_FMT_DBL:
        case AV_SAMPLE_FMT_DBLP:
            // double => reinterpret and then static cast down
            ret = STATIC_CAST(float, *REINTERPRET_CAST(double*, &val));
            break;

        default:
            fprintf(stderr, "Invalid sample format %s.\n", av_get_sample_fmt_name(codecCtx->sample_fmt));
            return 0;
    }

    return ret;
}

/**
 * Write the frame to an output file.
 */
void handleFrame(const AVCodecContext* codecCtx, const AVFrame* frame) {
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
}

/**
 * Find the first audio stream and returns its index. If there is no audio stream returns -1.
 */
int findAudioStream(const AVFormatContext* formatCtx) {
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
}

/*
 * Print information about the input file and the used codec.
 */
void printStreamInformation(const AVCodec* codec, const AVCodecContext* codecCtx, int audioStreamIndex) {
    fprintf(stderr, "Codec: %s\n", codec->long_name);
    if(codec->sample_fmts != NULL) {
        fprintf(stderr, "Supported sample formats: ");
        for(int i = 0; codec->sample_fmts[i] != -1; ++i) {
            fprintf(stderr, "%s", av_get_sample_fmt_name(codec->sample_fmts[i]));
            if(codec->sample_fmts[i+1] != -1) {
                fprintf(stderr, ", ");
            }
        }
        fprintf(stderr, "\n");
    }
    fprintf(stderr, "---------\n");
    fprintf(stderr, "Stream:        %7d\n", audioStreamIndex);
    fprintf(stderr, "Sample Format: %7s\n", av_get_sample_fmt_name(codecCtx->sample_fmt));
    fprintf(stderr, "Sample Rate:   %7d\n", codecCtx->sample_rate);
    fprintf(stderr, "Sample Size:   %7d\n", av_get_bytes_per_sample(codecCtx->sample_fmt));
    fprintf(stderr, "Channels:      %7d\n", codecCtx->channels);
    fprintf(stderr, "Planar:        %7d\n", av_sample_fmt_is_planar(codecCtx->sample_fmt));
    fprintf(stderr, "Float Output:  %7s\n", !RAW_OUT_ON_PLANAR || av_sample_fmt_is_planar(codecCtx->sample_fmt) ? "yes" : "no");
}

/**
 * Receive as many frames as available and handle them.
 */
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
