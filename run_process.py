import subprocess

# for i in [5, 8, 11, 14, 17, 20, 23, 26, 29, 30, 40, 50, 60, 70, 80, 90, 100]:
for i in ["F480M", "F430M", "F380M"]:
    subprocess.run(["python", "ngc1068_fitting.py", str(i)])
    # print(i)
