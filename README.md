# Use eIQ® toolkit to train model and deploy it on i.MX 93

## Quick Start

```bash
# clone the repo
$ git clone https://github.com/EricChen0313/eIQ-Project

# unzip the folder and transfer all the files to i.MX 93
# can refer slide 17

# install the packages you may need
$ pip3 install -r requirements.txt

# run the model with CPU
$ python3 run_model.py --model=<YOUR MODEL NAME> --image=<IMAGE NAME>

# run the model with NPU
$ python3 run_model.py --model=<YOUR MODEL NAME> --image=<IMAGE NAME> --use_npu
```

