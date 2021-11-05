import requests
import json

freqs = list()
words = list(map(str.strip, open("w.txt")))
cl = 1000
for i in range(0, len(words), cl):
    chunk = words[i:i+cl]
    r=requests.post("http://bos.zrc-sazu.si/cgi_new/ada.exe",
                    dict(name="dol_fre.cfg",
                         file="\r\n".join(chunk).encode("windows-1250")))
    content = r.content.decode("windows-1250")
    lines = content.split("</h2>")[1].split("\r\n<br>\r\n")[0].strip().splitlines()
    for line in lines:
        parts = line.split(" - ")
        if len(parts) == 2:
            f = dict(w=parts[0], f=int(parts[1][:-4]))
            print(f)
            freqs.append(f)



with open("freqs.json", "a") as f:
    json.dump(freqs, f)
