print("#!/bin/bash")

for i in range(1,41):
    SNRs = str(i).zfill(2)
    matname = "deepSNR" + SNRs + ".mat"
    part1 = "aws s3api copy-object --bucket SNR-bucket --key "
    part2 = " --copy-source my-bucket-test/"
    part3 = " --endpoint-url https://s3-west.nrp-nautilus.io"
    out = part1 + matname + part2 + matname + part3
    print(out)

print("aws s3api list-objects --bucket SNR-bucket  --endpoint-url https://s3-west.nrp-nautilus.io")

