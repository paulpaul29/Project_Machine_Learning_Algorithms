import subprocess
# vibrational levels of N2
for i in range(0, 46):
  #subprocess.call(['python', './src/regression_rec.py', f'{i}'])
  subprocess.call(['python', './src/regression_dis.py', f'{i}'])