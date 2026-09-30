from TTS.api import TTS

# Carica il modello
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# Testo che il robot dirà
testo_da_convertire = "Ciao, benvenuto. io sono la guida del museo"

output_file_name = "daEliminare"

# Genera audio con la tua voce
tts.tts_to_file(
    text=testo_da_convertire,
    file_path=output_file_name + ".wav",
    speaker_wav="voce_riferimento_1.wav",
    language="it"
)

print("Audio generato:" + output_file_name + ".wav")