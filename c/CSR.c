#include <vosk_api.h>
#include <stdio.h>

int main() {
    FILE *wavin;
    char buf[3200];
    int nread, final;

    VoskModel *model = vosk_model_new("model");
    VoskRecognizer *recognizer = vosk_recognizer_new(model, 16000.0);

    vosk_recognizer_dir(recognizer, "config.ini");    

    

    vosk_recognizer_free(recognizer);
    vosk_model_free(model);
    
    return 0;
}
