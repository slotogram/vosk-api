FF_ROOT?=/media/nti/Data970/code/FFmpeg
CFLAGS=-I../src
#LDFLAGS=-L../src -L$(FF_ROOT)/libavcodec -L$(FF_ROOT)/libavformat -L$(FF_ROOT)/libavutil -L$(FF_ROOT)/libswresample -lvosk -lavformat -lavcodec -lavutil -lswresample -ldl -lpthread -Wl,-rpath,../src
LDFLAGS=-L../src -lvosk -ldl -lpthread -Wl,-rpath,../src -Wl,-Bsymbolic

all: CSR test_vosk test_vosk_speaker

CSR: CSR.o
	gcc $^ -o $@ $(LDFLAGS)

test_vosk: test_vosk.o
	gcc $^ -o $@ $(LDFLAGS)

test_vosk_speaker: test_vosk_speaker.o
	gcc $^ -o $@ $(LDFLAGS)

%.o: %.c
	gcc $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.a CSR test_vosk test_vosk_speaker
