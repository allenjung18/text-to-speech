import tarfile
import urllib.request
import os

url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
output_dir = "train/smoke/LJSpeech"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "wavs"), exist_ok=True)

print("Starting download & extract...")
stream = urllib.request.urlopen(url)
tar = tarfile.open(fileobj=stream, mode="r|bz2")

extracted_wavs = 0
for member in tar:
    if member.name.endswith("metadata.csv"):
        print(f"Extracting {member.name}")
        tar.extract(member, output_dir)
    elif member.name.endswith(".wav"):
        if extracted_wavs < 200:
            tar.extract(member, output_dir)
            extracted_wavs += 1
            if extracted_wavs % 50 == 0:
                print(f"Extracted {extracted_wavs} wavs...")
        
    if extracted_wavs >= 200:
        # Check if we also have metadata.csv
        if os.path.exists(os.path.join(output_dir, "LJSpeech-1.1", "metadata.csv")):
            print("Finished extracting 200 wavs and metadata.")
            break
tar.close()
stream.close()
print("Done.")
