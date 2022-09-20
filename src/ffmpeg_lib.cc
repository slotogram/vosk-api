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


#include "ffmpeg_lib.h"


/**
 * Open an input file and the required decoder.
 * @param      filename             File to be opened
 * @param[out] input_format_context Format context of opened file
 * @param[out] input_codec_context  Codec context of opened file
 * @return Error code (0 if successful)
 */
int open_input_file(const char *filename,
                           AVFormatContext **input_format_context,
                           AVCodecContext **input_codec_context)
{
    AVCodecContext *avctx;
    const AVCodec *input_codec;
    const AVStream *stream;
    int error;

    /* Open the input file to read from it. */
    if ((error = avformat_open_input(input_format_context, filename, NULL,
                                     NULL)) < 0) {
        fprintf(stderr, "Could not open input file '%s' (error '%s')\n",
                filename, av_err2str(error));
        *input_format_context = NULL;
        return error;
    }

    /* Get information on the input file (number of streams etc.). */
    if ((error = avformat_find_stream_info(*input_format_context, NULL)) < 0) {
        fprintf(stderr, "Could not open find stream info (error '%s')\n",
                av_err2str(error));
        avformat_close_input(input_format_context);
        return error;
    }

    /* Make sure that there is only one stream in the input file. */
    if ((*input_format_context)->nb_streams != 1) {
        fprintf(stderr, "Expected one audio input stream, but found %d\n",
                (*input_format_context)->nb_streams);
        avformat_close_input(input_format_context);
        return AVERROR_EXIT;
    }

    stream = (*input_format_context)->streams[0];

    /* Find a decoder for the audio stream. */
    if (!(input_codec = avcodec_find_decoder(stream->codecpar->codec_id))) {
        fprintf(stderr, "Could not find input codec\n");
        avformat_close_input(input_format_context);
        return AVERROR_EXIT;
    }

    /* Allocate a new decoding context. */
    avctx = avcodec_alloc_context3(input_codec);
    if (!avctx) {
        fprintf(stderr, "Could not allocate a decoding context\n");
        avformat_close_input(input_format_context);
        return AVERROR(ENOMEM);
    }

    /* Initialize the stream parameters with demuxer information. */
    error = avcodec_parameters_to_context(avctx, stream->codecpar);
    if (error < 0) {
        avformat_close_input(input_format_context);
        avcodec_free_context(&avctx);
        return error;
    }

    /* Open the decoder for the audio stream to use it later. */
    if ((error = avcodec_open2(avctx, input_codec, NULL)) < 0) {
        fprintf(stderr, "Could not open input codec (error '%s')\n",
                av_err2str(error));
        avcodec_free_context(&avctx);
        avformat_close_input(input_format_context);
        return error;
    }

    /* Set the packet timebase for the decoder. */
    avctx->pkt_timebase = stream->time_base;

    /* Save the decoder context for easier access later. */
    *input_codec_context = avctx;

    return 0;
}

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
                            AVCodecContext **output_codec_context)
{
    AVCodecContext *avctx          = NULL;
    AVIOContext *output_io_context = NULL;
    AVStream *stream               = NULL;
    const AVCodec *output_codec    = NULL;
    int error;

    /* Open the output file to write to it. */
    if ((error = avio_open(&output_io_context, filename,
                           AVIO_FLAG_WRITE)) < 0) {
        fprintf(stderr, "Could not open output file '%s' (error '%s')\n",
                filename, av_err2str(error));
        return error;
    }

    /* Create a new format context for the output container format. */
    if (!(*output_format_context = avformat_alloc_context())) {
        fprintf(stderr, "Could not allocate output format context\n");
        return AVERROR(ENOMEM);
    }

    /* Associate the output file (pointer) with the container format context. */
    (*output_format_context)->pb = output_io_context;

    /* Guess the desired container format based on the file extension. */
    if (!((*output_format_context)->oformat = av_guess_format(NULL, filename,
                                                              NULL))) {
        fprintf(stderr, "Could not find output file format\n");
        goto cleanup;
    }

    if (!((*output_format_context)->url = av_strdup(filename))) {
        fprintf(stderr, "Could not allocate url.\n");
        error = AVERROR(ENOMEM);
        goto cleanup;
    }

    /* Find the encoder to be used by its name. */
    if (!(output_codec = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE))) {
        fprintf(stderr, "Could not find an AAC encoder.\n");
        goto cleanup;
    }

    /* Create a new audio stream in the output file container. */
    if (!(stream = avformat_new_stream(*output_format_context, NULL))) {
        fprintf(stderr, "Could not create new stream\n");
        error = AVERROR(ENOMEM);
        goto cleanup;
    }

    avctx = avcodec_alloc_context3(output_codec);
    if (!avctx) {
        fprintf(stderr, "Could not allocate an encoding context\n");
        error = AVERROR(ENOMEM);
        goto cleanup;
    }

    /* Set the basic encoder parameters.
     * The input file's sample rate is used to avoid a sample rate conversion. */
    av_channel_layout_default(&avctx->ch_layout, OUTPUT_CHANNELS);
    avctx->sample_rate    = sample_frequency;
    avctx->sample_fmt     = output_codec->sample_fmts[0];
    //avctx->bit_rate       = OUTPUT_BIT_RATE;

    /* Set the sample rate for the container. */
    stream->time_base.den = sample_frequency;
    stream->time_base.num = 1;

    /* Some container formats (like MP4) require global headers to be present.
     * Mark the encoder so that it behaves accordingly. */
    if ((*output_format_context)->oformat->flags & AVFMT_GLOBALHEADER)
        avctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    /* Open the encoder for the audio stream to use it later. */
    if ((error = avcodec_open2(avctx, output_codec, NULL)) < 0) {
        fprintf(stderr, "Could not open output codec (error '%s')\n",
                av_err2str(error));
        goto cleanup;
    }

    error = avcodec_parameters_from_context(stream->codecpar, avctx);
    if (error < 0) {
        fprintf(stderr, "Could not initialize stream parameters\n");
        goto cleanup;
    }

    /* Save the encoder context for easier access later. */
    *output_codec_context = avctx;

    return 0;

cleanup:
    avcodec_free_context(&avctx);
    avio_closep(&(*output_format_context)->pb);
    avformat_free_context(*output_format_context);
    *output_format_context = NULL;
    return error < 0 ? error : AVERROR_EXIT;
}

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
                            AVCodecContext **output_codec_context)
{
    AVCodecContext *avctx          = NULL;
    const AVCodec *output_codec    = NULL;
    int error;


    /* Find the encoder to be used by its name. */
    if (!(output_codec = avcodec_find_encoder(AV_CODEC_ID_PCM_S16LE))) {
        fprintf(stderr, "Could not find an AAC encoder.\n");
        goto cleanup;
    }

    avctx = avcodec_alloc_context3(output_codec);
    if (!avctx) {
        fprintf(stderr, "Could not allocate an encoding context\n");
        error = AVERROR(ENOMEM);
        goto cleanup;
    }

    /* Set the basic encoder parameters.
     * The input file's sample rate is used to avoid a sample rate conversion. */
    av_channel_layout_default(&avctx->ch_layout, OUTPUT_CHANNELS);
    avctx->sample_rate    = sample_frequency;
    avctx->sample_fmt     = output_codec->sample_fmts[0];
    //avctx->bit_rate       = OUTPUT_BIT_RATE;

  
    /* Open the encoder for the audio stream to use it later. */
    if ((error = avcodec_open2(avctx, output_codec, NULL)) < 0) {
        fprintf(stderr, "Could not open output codec (error '%s')\n",
                av_err2str(error));
        goto cleanup;
    }

    /* Save the encoder context for easier access later. */
    *output_codec_context = avctx;

    return 0;

cleanup:
    avcodec_free_context(&avctx);
    return error < 0 ? error : AVERROR_EXIT;
}



/**
 * Initialize one data packet for reading or writing.
 * @param[out] packet Packet to be initialized
 * @return Error code (0 if successful)
 */
int init_packet(AVPacket **packet)
{
    if (!(*packet = av_packet_alloc())) {
        fprintf(stderr, "Could not allocate packet\n");
        return AVERROR(ENOMEM);
    }
    return 0;
}

/**
 * Initialize one audio frame for reading from the input file.
 * @param[out] frame Frame to be initialized
 * @return Error code (0 if successful)
 */
int init_input_frame(AVFrame **frame)
{
    if (!(*frame = av_frame_alloc())) {
        fprintf(stderr, "Could not allocate input frame\n");
        return AVERROR(ENOMEM);
    }
    return 0;
}

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
                          SwrContext **resample_context)
{
        int error;

        /*
         * Create a resampler context for the conversion.
         * Set the conversion parameters.
         */
        error = swr_alloc_set_opts2(resample_context,
                                             &output_codec_context->ch_layout,
                                              output_codec_context->sample_fmt,
                                              output_codec_context->sample_rate,
                                             &input_codec_context->ch_layout,
                                              input_codec_context->sample_fmt,
                                              input_codec_context->sample_rate,
                                              0, NULL);
        if (error < 0) {
            fprintf(stderr, "Could not allocate resample context\n");
            return error;
        }
        /*
        * Perform a sanity check so that the number of converted samples is
        * not greater than the number of samples to be converted.
        * If the sample rates differ, this case has to be handled differently
        */
        //av_assert0(output_codec_context->sample_rate == input_codec_context->sample_rate);

        /* Open the resampler with the specified parameters. */
        if ((error = swr_init(*resample_context)) < 0) {
            fprintf(stderr, "Could not open resample context\n");
            swr_free(resample_context);
            return error;
        }
    return 0;
}

/**
 * Initialize a FIFO buffer for the audio samples to be encoded.
 * @param[out] fifo                 Sample buffer
 * @param      output_codec_context Codec context of the output file
 * @return Error code (0 if successful)
 */
int init_fifo(AVAudioFifo **fifo, AVCodecContext *output_codec_context)
{
    /* Create the FIFO buffer based on the specified output sample format. */
    if (!(*fifo = av_audio_fifo_alloc(output_codec_context->sample_fmt,
                                      output_codec_context->ch_layout.nb_channels, 1))) {
        fprintf(stderr, "Could not allocate FIFO\n");
        return AVERROR(ENOMEM);
    }
    return 0;
}

/**
 * Write the header of the output file container.
 * @param output_format_context Format context of the output file
 * @return Error code (0 if successful)
 */
int write_output_file_header(AVFormatContext *output_format_context)
{
    int error;
    if ((error = avformat_write_header(output_format_context, NULL)) < 0) {
        fprintf(stderr, "Could not write output file header (error '%s')\n",
                av_err2str(error));
        return error;
    }
    return 0;
}

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
                              int *data_present, int *finished)
{
    /* Packet used for temporary storage. */
    AVPacket *input_packet;
    int error;

    error = init_packet(&input_packet);
    if (error < 0)
        return error;
    *data_present = 0;
    	*finished = 0;
    /* Read one audio frame from the input file into a temporary packet. */
    if ((error = av_read_frame(input_format_context, input_packet)) < 0) {
        /* If we are at the end of the file, flush the decoder below. */
        if (error == AVERROR_EOF) {
            *finished = 1; }
        else {
            fprintf(stderr, "Could not read frame (error '%s')\n",
                    av_err2str(error));
            goto cleanup;
        }
    }
    /* Send the audio frame stored in the temporary packet to the decoder.
     * The input audio stream decoder is used to do this. */
    if ((error = avcodec_send_packet(input_codec_context, input_packet)) < 0) {
        fprintf(stderr, "Could not send packet for decoding (error '%s')\n",
                av_err2str(error));
        goto cleanup;
    }
    /* Receive one frame from the decoder. */
    error = avcodec_receive_frame(input_codec_context, frame);

    /* If the decoder asks for more data to be able to decode a frame,
     * return indicating that no data is present. */
    if (error == AVERROR(EAGAIN)) {
        error = 0;
        goto cleanup;
    /* If the end of the input file is reached, stop decoding. */
    } else if (error == AVERROR_EOF) {
        *finished = 1;
        error = 0;
        goto cleanup;
    } else if (error < 0) {
        fprintf(stderr, "Could not decode frame (error '%s')\n",
                av_err2str(error));
        goto cleanup;
    /* Default case: Return decoded data. */
    } else {
        *data_present = 1;
        goto cleanup;
    }

cleanup:
	
	if (input_packet)
	    av_packet_free(&input_packet);
    return error;
}

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
                                  int frame_size)
{
    int error;

    /* Allocate as many pointers as there are audio channels.
     * Each pointer will later point to the audio samples of the corresponding
     * channels (although it may be NULL for interleaved formats).
     */
    if (!(*converted_input_samples = (uint8_t**)calloc(output_codec_context->ch_layout.nb_channels,
                                            sizeof(**converted_input_samples)))) {
        fprintf(stderr, "Could not allocate converted input sample pointers\n");
        return AVERROR(ENOMEM);
    }

    /* Allocate memory for the samples of all channels in one consecutive
     * block for convenience. */
    if ((error = av_samples_alloc(*converted_input_samples, NULL,
                                  output_codec_context->ch_layout.nb_channels,
                                  frame_size,
                                  output_codec_context->sample_fmt, 0)) < 0) {
        fprintf(stderr,
                "Could not allocate converted input samples (error '%s')\n",
                av_err2str(error));
        av_freep(&(*converted_input_samples)[0]);
        free(*converted_input_samples);
        return error;
    }
    return 0;
}

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
                           SwrContext *resample_context)
{
    int error;

    /* Convert the samples using the resampler. */
    if ((error = swr_convert(resample_context,
                             converted_data, out_frame_size,
                             input_data    , frame_size)) < 0) {
        fprintf(stderr, "Could not convert input samples (error '%s')\n",
                av_err2str(error));
        return error;
    }

    return 0;
}

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
                               const int frame_size)
{
    int error;

    /* Make the FIFO as large as it needs to be to hold both,
     * the old and the new samples. */
    if ((error = av_audio_fifo_realloc(fifo, av_audio_fifo_size(fifo) + frame_size)) < 0) {
        fprintf(stderr, "Could not reallocate FIFO\n");
        return error;
    }

    /* Store the new samples in the FIFO buffer. */
    if (av_audio_fifo_write(fifo, (void **)converted_input_samples,
                            frame_size) < frame_size) {
        fprintf(stderr, "Could not write data to FIFO\n");
        return AVERROR_EXIT;
    }
    return 0;
}

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
                                         int *finished)
{	
    /* Temporary storage of the input samples of the frame read from the file. */
    AVFrame *input_frame = NULL;
    /* Temporary storage for the converted input samples. */
    uint8_t **converted_input_samples = NULL;
    int data_present;
    int ret = AVERROR_EXIT;

    /* Initialize temporary storage for one input frame. */
    if (init_input_frame(&input_frame))
        goto cleanup;
    /* Decode one frame worth of audio samples. */
    if (decode_audio_frame(input_frame, input_format_context,
                           input_codec_context, &data_present, finished))
        goto cleanup;
    /* If we are at the end of the file and there are no more samples
     * in the decoder which are delayed, we are actually finished.
     * This must not be treated as an error. */
    if (*finished) {
        ret = 0;
        goto cleanup;
    }
    /* If there is decoded data, convert and store it. */
    if (data_present) {

    	int nb_samples = input_frame->nb_samples;
	if (input_codec_context->sample_rate != output_codec_context->sample_rate) 
		nb_samples = av_rescale_rnd(swr_get_delay(resampler_context, input_codec_context->sample_rate) +
                                     input_frame->nb_samples, output_codec_context->sample_rate, input_codec_context->sample_rate, AV_ROUND_UP);

        /* Initialize the temporary storage for the converted input samples. */
        if (init_converted_samples(&converted_input_samples, output_codec_context,
                                   nb_samples))
            goto cleanup;

    	if (convert_samples((const uint8_t**)input_frame->extended_data, converted_input_samples,
                            input_frame->nb_samples, nb_samples, resampler_context))
            goto cleanup;

        /* Add the converted input samples to the FIFO buffer for later processing. */
        if (add_samples_to_fifo(fifo, converted_input_samples,
                                nb_samples))
            goto cleanup;
        ret = 0;

    }
    ret = 0;

cleanup:
    if (converted_input_samples) {
        av_freep(&converted_input_samples[0]);
        free(converted_input_samples);
    }
    av_frame_free(&input_frame);

    return ret;
}

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
                             int frame_size)
{
    int error;

    /* Create a new frame to store the audio samples. */
    if (!(*frame = av_frame_alloc())) {
        fprintf(stderr, "Could not allocate output frame\n");
        return AVERROR_EXIT;
    }

    /* Set the frame's parameters, especially its size and format.
     * av_frame_get_buffer needs this to allocate memory for the
     * audio samples of the frame.
     * Default channel layouts based on the number of channels
     * are assumed for simplicity. */
    (*frame)->nb_samples     = frame_size;
    av_channel_layout_copy(&(*frame)->ch_layout, &output_codec_context->ch_layout);
    (*frame)->format         = output_codec_context->sample_fmt;
    (*frame)->sample_rate    = output_codec_context->sample_rate;
    //fprintf(stderr, "frame_size'%d')\n", frame_size);

    /* Allocate the samples of the created frame. This call will make
     * sure that the audio frame can hold as many samples as specified. */
    if ((error = av_frame_get_buffer(*frame, 0)) < 0) {
        fprintf(stderr, "Could not allocate output frame samples (error '%s')\n",
                av_err2str(error));
        av_frame_free(frame);
        return error;
    }

    return 0;
}


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
                              int *data_present)
{
    /* Packet used for temporary storage. */
    AVPacket *output_packet;
    int error;

    error = init_packet(&output_packet);
    if (error < 0)
        return error;

    /* Set a timestamp based on the sample rate for the container. */
    if (frame) {
        frame->pts = pts;
        pts += frame->nb_samples;
    }

    *data_present = 0;
    /* Send the audio frame stored in the temporary packet to the encoder.
     * The output audio stream encoder is used to do this. */
    error = avcodec_send_frame(output_codec_context, frame);
    /* Check for errors, but proceed with fetching encoded samples if the
     *  encoder signals that it has nothing more to encode. */
    if (error < 0 && error != AVERROR_EOF) {
      fprintf(stderr, "Could not send packet for encoding (error '%s')\n",
              av_err2str(error));
      goto cleanup;
    }

    /* Receive one encoded frame from the encoder. */
    error = avcodec_receive_packet(output_codec_context, output_packet);
    /* If the encoder asks for more data to be able to provide an
     * encoded frame, return indicating that no data is present. */
    if (error == AVERROR(EAGAIN)) {
        error = 0;
        goto cleanup;
    /* If the last frame has been encoded, stop encoding. */
    } else if (error == AVERROR_EOF) {
        error = 0;
        goto cleanup;
    } else if (error < 0) {
        fprintf(stderr, "Could not encode frame (error '%s')\n",
                av_err2str(error));
        goto cleanup;
    /* Default case: Return encoded data. */
    } else {
        *data_present = 1;
    }

    /* Write one audio frame from the temporary packet to the output file. */
    if (*data_present &&
        (error = av_write_frame(output_format_context, output_packet)) < 0) {
        fprintf(stderr, "Could not write frame (error '%s')\n",
                av_err2str(error));
        goto cleanup;
    }

cleanup:
    av_packet_free(&output_packet);
    return error;
}

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
                                 AVCodecContext *output_codec_context)
{
    /* Temporary storage of the output samples of the frame written to the file. */
    AVFrame *output_frame;
    /* Use the maximum number of possible samples per frame.
     * If there is less than the maximum possible frame size in the FIFO
     * buffer use this number. Otherwise, use the maximum possible frame size. */
//    const int frame_size = FFMIN(av_audio_fifo_size(fifo), output_codec_context->frame_size);
    const int frame_size = av_audio_fifo_size(fifo);
    int data_written;

    /* Initialize temporary storage for one output frame. */
    if (init_output_frame(&output_frame, output_codec_context, frame_size))
        return AVERROR_EXIT;

    /* Read as many samples from the FIFO buffer as required to fill the frame.
     * The samples are stored in the frame temporarily. */
    if (av_audio_fifo_read(fifo, (void **)output_frame->data, frame_size) < frame_size) {
        fprintf(stderr, "Could not read data from FIFO\n");
        av_frame_free(&output_frame);
        return AVERROR_EXIT;
    }

    /* Encode one frame worth of audio samples. */
    if (encode_audio_frame(output_frame, output_format_context,
                           output_codec_context, &data_written)) {
        av_frame_free(&output_frame);
        return AVERROR_EXIT;
    }
    av_frame_free(&output_frame);
    return 0;
}

/**
 * Load one audio frame from the FIFO buffer, encode and write it to the
 * output file.
 * @param fifo                  Buffer used for temporary storage
 * @param output_format_context Format context of the output file
 * @param output_codec_context  Codec context of the output file
 * @return Error code (0 if successful)
 */
int load_encoded_samples(AVAudioFifo *fifo,
                                 AVFormatContext *output_format_context,
                                 AVCodecContext *output_codec_context)
{
    /* Temporary storage of the output samples of the frame written to the file. */
    AVFrame *output_frame;
    /* Use the maximum number of possible samples per frame.
     * If there is less than the maximum possible frame size in the FIFO
     * buffer use this number. Otherwise, use the maximum possible frame size. */
//    const int frame_size = FFMIN(av_audio_fifo_size(fifo), output_codec_context->frame_size);
    const int frame_size = av_audio_fifo_size(fifo);
    int data_written;

    /* Initialize temporary storage for one output frame. */
    if (init_output_frame(&output_frame, output_codec_context, frame_size))
        return AVERROR_EXIT;

    /* Read as many samples from the FIFO buffer as required to fill the frame.
     * The samples are stored in the frame temporarily. */
    if (av_audio_fifo_read(fifo, (void **)output_frame->data, frame_size) < frame_size) {
        fprintf(stderr, "Could not read data from FIFO\n");
        av_frame_free(&output_frame);
        return AVERROR_EXIT;
    }

//return link to 
    /* Encode one frame worth of audio samples. */
    if (encode_audio_frame(output_frame, output_format_context,
                           output_codec_context, &data_written)) {
        av_frame_free(&output_frame);
        return AVERROR_EXIT;
    }
    av_frame_free(&output_frame);
    return 0;
}

/**
 * Write the trailer of the output file container.
 * @param output_format_context Format context of the output file
 * @return Error code (0 if successful)
 */
int write_output_file_trailer(AVFormatContext *output_format_context)
{
    int error;
    if ((error = av_write_trailer(output_format_context)) < 0) {
        fprintf(stderr, "Could not write output file trailer (error '%s')\n",
                av_err2str(error));
        return error;
    }
    return 0;
}



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
