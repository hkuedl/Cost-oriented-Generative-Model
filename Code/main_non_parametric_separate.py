{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ca61d734",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import os\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "from torch.utils.data import TensorDataset, DataLoader\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "import math\n",
    "from utils import *\n",
    "import copy\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from benchmark_non_parametric import *\n",
    "from data_loader import Dataset_load_single_node_parametric,Dataset_load_single_node_non_parametric\n",
    "from combined_data_loader import *\n",
    "from Optimization_single_node import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bbefc0df",
   "metadata": {},
   "outputs": [],
   "source": [
    "data=pd.read_csv('../Data/load_data_city_4_2.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "3cbaece2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import time\n",
    "import numpy as np\n",
    "import torch\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "class Args:\n",
    "    def __init__(self):\n",
    "        # ----- problem / optnet -----\n",
    "        self.T = 24\n",
    "        self.base_mva = 100.0\n",
    "        self.capacity_scale = 4.5\n",
    "        self.ramp_rate = 0.5\n",
    "        self.voll = 200.0\n",
    "        self.vosp = 50.0\n",
    "        self.M_beta = 1e4\n",
    "        self.N_scen = 10\n",
    "        self.pwl_segments = 10\n",
    "\n",
    "        # IMPORTANT: add these to match gurobi\n",
    "        self.reserve_up_ratio = 0.05\n",
    "        self.reserve_dn_ratio = 0.02\n",
    "        self.rt_up_ratio = 3.0\n",
    "        self.rt_dn_ratio = 0.5\n",
    "\n",
    "        # ----- training -----\n",
    "        self.device = \"cuda\"\n",
    "        self.epochs = 1\n",
    "        self.train_batch_size = 8\n",
    "        self.test_batch_size = 8\n",
    "        self.lr = 1e-7\n",
    "        self.solver = \"ECOS\"\n",
    "\n",
    "        self.N_scen = 20       # <== OptNet真正求解的场景池 (即 K)\n",
    "        self.S_full = 200       # VAE 现场吐出的大量候选场景数 (S 池)\n",
    "        self.K_rand = 10       # K里面有多少条纯随机保留(防过拟合)\n",
    "        self.tau_gumbel = 1.0     # Gumbel Softmax 温度\n",
    "        self.eps_uniform = 0.1 # 防震荡平滑参数\n",
    "        self.lambda_div = 1e5   # [新增] 避免多头选到同一个场景的相互排斥惩罚力度\n",
    "\n",
    "        self.filter_epochs = 5 # Stage 2 (训Filter) 轮数\n",
    "        self.filter_lr = 1e-3   # Stage 2 学习率\n",
    "        self.dfl_epochs = 1     # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "        self.dfl_lr = 1e-6      # Stage 3 学习率 (必须极小，防崩坏)\n",
    "        self.eval_mode = \"discrete\"\n",
    "\n",
    "args=Args()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "24b3cb26",
   "metadata": {},
   "outputs": [],
   "source": [
    "set_seed(42)\n",
    "data_path = \"../Data/load_data_city_4_2.csv\"\n",
    "quantiles = [0.05*i for i in range(1,20)]\n",
    "\n",
    "eps_search=pd.read_csv('../Result/eps_search.csv')\n",
    "eps=int(eps_search[eps_search['model']=='non_parametric']['eps'])\n",
    "target_nodes = [f\"4-2-{i}\" for i in range(11)]\n",
    "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
    "\n",
    "data=pd.read_csv(data_path)\n",
    "data[\"DATETIME\"] = pd.to_datetime(data[\"DATETIME\"], errors=\"coerce\")\n",
    "data_2022 = data[data[\"DATETIME\"].dt.year == 2022].copy()\n",
    "Lmin, Lmax = system_hourly_load_minmax(data_2022, datetime_col=\"DATETIME\",node_cols=target_nodes)\n",
    "Lmax_total=Lmax.sum(0)# (24,)\n",
    "Lmin_total=Lmin.sum(0) # (24,)\n",
    "args.Lmax_total=Lmax_total\n",
    "args.Lmin_total=Lmin_total\n",
    "args.eps_value=eps"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2c661ede",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch    1 | train=0.278340 | val=0.123551\n"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Input \u001b[0;32mIn [5]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[0;32m----> 1\u001b[0m models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric \u001b[38;5;241m=\u001b[39m \u001b[43mrun_non_parametric_benchmark\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m      2\u001b[0m \u001b[43m    \u001b[49m\u001b[43mDataHandler\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mDataset_load_single_node_non_parametric\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      3\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdata_path\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdata_path\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      4\u001b[0m \u001b[43m    \u001b[49m\u001b[43mdevice\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdevice\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      5\u001b[0m \u001b[43m    \u001b[49m\u001b[43mepochs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m1000\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m      6\u001b[0m \u001b[43m    \u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m32\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m      7\u001b[0m \u001b[43m    \u001b[49m\u001b[43mtarget_nodes\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtarget_nodes\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m      8\u001b[0m \u001b[43m    \u001b[49m\u001b[43mlr\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m1e-4\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m      9\u001b[0m \u001b[43m    \u001b[49m\u001b[43mverbose\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m     10\u001b[0m \u001b[43m    \u001b[49m\u001b[43mtrain_length\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m8760\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m     11\u001b[0m \u001b[43m    \u001b[49m\u001b[43mval_ratio\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m0.2\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m     12\u001b[0m \u001b[43m    \u001b[49m\u001b[43mseed\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;241;43m42\u001b[39;49m\u001b[43m,\u001b[49m\n\u001b[1;32m     13\u001b[0m \u001b[43m    \u001b[49m\u001b[43mquantiles\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mquantiles\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m     14\u001b[0m \u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/Meta-DFL/Code joint forecast copy 5/benchmark_non_parametric.py:165\u001b[0m, in \u001b[0;36mrun_non_parametric_benchmark\u001b[0;34m(DataHandler, device, epochs, batch_size, target_nodes, lr, hidden, patience, ckpt_dir, verbose, data_path, train_length, val_ratio, seed, quantiles)\u001b[0m\n\u001b[1;32m    160\u001b[0m runner \u001b[38;5;241m=\u001b[39m Runner_non_parametric(\n\u001b[1;32m    161\u001b[0m     train_set, val_set, test_set,\n\u001b[1;32m    162\u001b[0m     quantiles\u001b[38;5;241m=\u001b[39mquantiles, hidden\u001b[38;5;241m=\u001b[39mhidden, lr\u001b[38;5;241m=\u001b[39mlr, device\u001b[38;5;241m=\u001b[39mdevice\n\u001b[1;32m    163\u001b[0m )\n\u001b[1;32m    164\u001b[0m best_path \u001b[38;5;241m=\u001b[39m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mjoin(ckpt_dir, \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbest_\u001b[39m\u001b[38;5;132;01m{\u001b[39;00mnode\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m.pt\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m--> 165\u001b[0m \u001b[43mrunner\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mfit\u001b[49m\u001b[43m(\u001b[49m\u001b[43mepochs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mepochs\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mbatch_size\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbatch_size\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mpatience\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mpatience\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mbest_path\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mbest_path\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mverbose\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mverbose\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    167\u001b[0m model \u001b[38;5;241m=\u001b[39m runner\u001b[38;5;241m.\u001b[39mmodel\u001b[38;5;241m.\u001b[39mto(device)\u001b[38;5;241m.\u001b[39meval()\n\u001b[1;32m    168\u001b[0m models_s[node] \u001b[38;5;241m=\u001b[39m model\n",
      "File \u001b[0;32m~/Meta-DFL/Code joint forecast copy 5/benchmark_non_parametric.py:82\u001b[0m, in \u001b[0;36mRunner_non_parametric.fit\u001b[0;34m(self, epochs, batch_size, patience, best_path, verbose)\u001b[0m\n\u001b[1;32m     80\u001b[0m loss \u001b[38;5;241m=\u001b[39m pinball_loss_non_parametric(q_hat, y, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mquantiles)\n\u001b[1;32m     81\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mopt\u001b[38;5;241m.\u001b[39mzero_grad()\n\u001b[0;32m---> 82\u001b[0m \u001b[43mloss\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbackward\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     83\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mopt\u001b[38;5;241m.\u001b[39mstep()\n\u001b[1;32m     84\u001b[0m tr_losses\u001b[38;5;241m.\u001b[39mappend(\u001b[38;5;28mfloat\u001b[39m(loss\u001b[38;5;241m.\u001b[39mitem()))\n",
      "File \u001b[0;32m~/.local/lib/python3.8/site-packages/torch/_tensor.py:492\u001b[0m, in \u001b[0;36mTensor.backward\u001b[0;34m(self, gradient, retain_graph, create_graph, inputs)\u001b[0m\n\u001b[1;32m    482\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m has_torch_function_unary(\u001b[38;5;28mself\u001b[39m):\n\u001b[1;32m    483\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m handle_torch_function(\n\u001b[1;32m    484\u001b[0m         Tensor\u001b[38;5;241m.\u001b[39mbackward,\n\u001b[1;32m    485\u001b[0m         (\u001b[38;5;28mself\u001b[39m,),\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m    490\u001b[0m         inputs\u001b[38;5;241m=\u001b[39minputs,\n\u001b[1;32m    491\u001b[0m     )\n\u001b[0;32m--> 492\u001b[0m \u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mautograd\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mbackward\u001b[49m\u001b[43m(\u001b[49m\n\u001b[1;32m    493\u001b[0m \u001b[43m    \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mgradient\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mretain_graph\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcreate_graph\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43minputs\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43minputs\u001b[49m\n\u001b[1;32m    494\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m~/.local/lib/python3.8/site-packages/torch/autograd/__init__.py:251\u001b[0m, in \u001b[0;36mbackward\u001b[0;34m(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)\u001b[0m\n\u001b[1;32m    246\u001b[0m     retain_graph \u001b[38;5;241m=\u001b[39m create_graph\n\u001b[1;32m    248\u001b[0m \u001b[38;5;66;03m# The reason we repeat the same comment below is that\u001b[39;00m\n\u001b[1;32m    249\u001b[0m \u001b[38;5;66;03m# some Python versions print out the first line of a multi-line function\u001b[39;00m\n\u001b[1;32m    250\u001b[0m \u001b[38;5;66;03m# calls in the traceback and some print out the last line\u001b[39;00m\n\u001b[0;32m--> 251\u001b[0m \u001b[43mVariable\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_execution_engine\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mrun_backward\u001b[49m\u001b[43m(\u001b[49m\u001b[43m  \u001b[49m\u001b[38;5;66;43;03m# Calls into the C++ engine to run the backward pass\u001b[39;49;00m\n\u001b[1;32m    252\u001b[0m \u001b[43m    \u001b[49m\u001b[43mtensors\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    253\u001b[0m \u001b[43m    \u001b[49m\u001b[43mgrad_tensors_\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    254\u001b[0m \u001b[43m    \u001b[49m\u001b[43mretain_graph\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    255\u001b[0m \u001b[43m    \u001b[49m\u001b[43mcreate_graph\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    256\u001b[0m \u001b[43m    \u001b[49m\u001b[43minputs\u001b[49m\u001b[43m,\u001b[49m\n\u001b[1;32m    257\u001b[0m \u001b[43m    \u001b[49m\u001b[43mallow_unreachable\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m    258\u001b[0m \u001b[43m    \u001b[49m\u001b[43maccumulate_grad\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[38;5;28;43;01mTrue\u001b[39;49;00m\u001b[43m,\u001b[49m\n\u001b[1;32m    259\u001b[0m \u001b[43m\u001b[49m\u001b[43m)\u001b[49m\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric = run_non_parametric_benchmark(\n",
    "    DataHandler=Dataset_load_single_node_non_parametric,\n",
    "    data_path=data_path,\n",
    "    device=device,\n",
    "    epochs=1000,\n",
    "    batch_size=32,\n",
    "    target_nodes=target_nodes,\n",
    "    lr=1e-4,\n",
    "    verbose=True,\n",
    "    train_length=8760,\n",
    "    val_ratio=0.2,\n",
    "    seed=42,\n",
    "    quantiles=quantiles,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "e93137f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "set_seed(0)\n",
    "window_pack_non_parametric_train = sample_window_non_parametric_benchmark(\n",
    "    models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric,\n",
    "    target_nodes=target_nodes, split=\"train\", horizon_days=292, start_day=0, n_samples=200, seq_len=24\n",
    ")\n",
    "\n",
    "set_seed(0)\n",
    "window_pack_non_parametric_val = sample_window_non_parametric_benchmark(\n",
    "    models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric,\n",
    "    target_nodes=target_nodes, split=\"val\", horizon_days=73, start_day=0, n_samples=200, seq_len=24\n",
    ")\n",
    "\n",
    "set_seed(0)\n",
    "window_pack_non_parametric_test = sample_window_non_parametric_benchmark(\n",
    "    models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric,\n",
    "    target_nodes=target_nodes, split=\"test\", horizon_days=303, start_day=0, n_samples=200, seq_len=24\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "2fe408e9",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'window_pack_full_non_parametric' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[0;32mIn [8]\u001b[0m, in \u001b[0;36m<cell line: 13>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m set_seed(\u001b[38;5;241m0\u001b[39m)\n\u001b[1;32m      2\u001b[0m window_pack_draw \u001b[38;5;241m=\u001b[39m sample_window_non_parametric_benchmark(\n\u001b[1;32m      3\u001b[0m     models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric,\n\u001b[1;32m      4\u001b[0m     target_nodes\u001b[38;5;241m=\u001b[39mtarget_nodes,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m      9\u001b[0m     seq_len\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m24\u001b[39m,\n\u001b[1;32m     10\u001b[0m )\n\u001b[0;32m---> 13\u001b[0m dfm \u001b[38;5;241m=\u001b[39m compute_metrics_window(\u001b[43mwindow_pack_full_non_parametric\u001b[49m)\n\u001b[1;32m     14\u001b[0m \u001b[38;5;28mprint\u001b[39m(dfm)\n\u001b[1;32m     15\u001b[0m plot_window_curve(window_pack_draw, print_metrics\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'window_pack_full_non_parametric' is not defined"
     ]
    }
   ],
   "source": [
    "set_seed(0)\n",
    "window_pack_draw = sample_window_non_parametric_benchmark(\n",
    "    models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    split=\"test\",      # NEW\n",
    "    horizon_days=3,\n",
    "    start_day=0,\n",
    "    n_samples=200,\n",
    "    seq_len=24,\n",
    ")\n",
    "\n",
    "\n",
    "dfm = compute_metrics_window(window_pack_full_non_parametric)\n",
    "print(dfm)\n",
    "plot_window_curve(window_pack_draw, print_metrics=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "e9e88e23",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('../Result/Non_parametric/models_s.pkl', 'wb') as f:\n",
    "    pickle.dump(models_s_non_parametric, f)\n",
    "with open('../Result/Non_parametric/handlers_s.pkl', 'wb') as f:\n",
    "    pickle.dump(handlers_s_non_parametric, f)\n",
    "with open('../Result/Non_parametric/pack_data_s.pkl', 'wb') as f:\n",
    "    pickle.dump(pack_data_s_non_parametric, f)\n",
    "\n",
    "with open('../Result/Non_parametric/window_pack_non_parametric_train.pkl', 'wb') as f:\n",
    "    pickle.dump(window_pack_non_parametric_train, f)\n",
    "with open('../Result/Non_parametric/window_pack_non_parametric_val.pkl', 'wb') as f:\n",
    "    pickle.dump(window_pack_non_parametric_val, f)\n",
    "with open('../Result/Non_parametric/window_pack_non_parametric_test.pkl', 'wb') as f:\n",
    "    pickle.dump(window_pack_non_parametric_test, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3b1f748e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "with open('../Result/Non_parametric/models_s.pkl', 'rb') as f:\n",
    "    models_s_non_parametric = pickle.load(f)\n",
    "with open('../Result/Non_parametric/handlers_s.pkl', 'rb') as f:\n",
    "    handlers_s_non_parametric = pickle.load(f)\n",
    "with open('../Result/Non_parametric/pack_data_s.pkl', 'rb') as f:\n",
    "    pack_data_s_non_parametric = pickle.load(f)\n",
    "\n",
    "with open('../Result/Non_parametric/window_pack_non_parametric_val.pkl', 'rb') as f:\n",
    "    window_pack_non_parametric_val = pickle.load(f)\n",
    "with open('../Result/Non_parametric/window_pack_non_parametric_train.pkl', 'rb') as f:\n",
    "    window_pack_non_parametric_train = pickle.load(f)\n",
    "with open('../Result/Non_parametric/window_pack_non_parametric_test.pkl', 'rb') as f:\n",
    "    window_pack_non_parametric_test = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c6e9e769",
   "metadata": {},
   "outputs": [],
   "source": [
    "optimization_mode = \"single\"\n",
    "problem_mode = \"dro\"\n",
    "forecasting_mode = \"separate\"\n",
    "\n",
    "# 构建 manager 主要为了拿拓扑或者映射\n",
    "if optimization_mode == \"multi\":\n",
    "    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "else:\n",
    "    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == \"multi\":\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "# 第一次完整跑 Stage A + Stage B\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.eval_mode = \"discrete\"\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 1e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full = 200\n",
    "args.lambda_div = 1e5\n",
    "args.lambda_div_stage3 = 1e5\n",
    "\n",
    "\n",
    "args.div_type = \"kl\"\n",
    "result_dro_single_kl = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(True, True, True, True, True),\n",
    "    stage2_artifact=None,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_single_kl,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/NonParametric/KL\"\n",
    ")\n",
    "\n",
    "compare_res_single_dro_kl = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_kl,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\", \"random\", \"kmeans\", \"kmedoids\", \"hierarchical\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "shared_stage2_artifact = result_dro_single_kl[\"stage2_artifact\"]\n",
    "args.div_type = 'entropy'\n",
    "result_dro_single_entropy = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(False, False, False, False, False),\n",
    "    stage2_artifact=shared_stage2_artifact,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(result_dro_single_entropy, forecasting_mode, out_dir=\"../Result/NonParametric/Entropy\")\n",
    "\n",
    "compare_res_single_dro_entropy = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_entropy,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "\n",
    "args.lambda_div = 5e8\n",
    "args.lambda_div_stage3 = 5e8\n",
    "args.div_type = 'inner'\n",
    "result_dro_single_inner = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(False, False, False, False, False),\n",
    "    stage2_artifact=shared_stage2_artifact,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(result_dro_single_inner, forecasting_mode, out_dir=\"../Result/NonParametric/Inner\")\n",
    "\n",
    "compare_res_single_dro_inner = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_inner,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "\n",
    "compare_res_single_dro_all = merge_compare_results_learned_as_variants(\n",
    "    compare_res_single_dro_kl,\n",
    "    compare_res_single_dro_inner,\n",
    "    compare_res_single_dro_entropy,\n",
    ")\n",
    "\n",
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_dro_all)\n",
    "for row in summary_rows:\n",
    "    print(row)\n",
    "\n",
    "with open('../Result/Non_parametric/compare_res_single_dro_all.pkl', 'wb') as f:\n",
    "    pickle.dump(compare_res_single_dro_all, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e81e11e1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "args.train_batch_size = 8\n",
      "args.test_batch_size = 8\n",
      "run_stage2: True run_stage3: True reused_stage2: False multi: False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:06<00:00,  5.72it/s, avg=4.26e+5, loss=2.69e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425584.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:55<00:00,  1.45s/it, avg=3.8e+5, loss=2.61e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "379849.5\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, train_bs=8)\n",
      "train_mode = dfl\n",
      "scenario_filter type = RandomScenarioSelector\n",
      "has_learnable_filter = False\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 0\n",
      "num predictor params = 401489\n",
      "train_batch_size = 8\n",
      "===== optimizer param groups =====\n",
      "group 0: name=predictor, lr=1e-06, n_params=401489\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  33%|███▎      | 1/3 [02:04<04:09, 124.73s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      " group 0: name=predictor, lr=1e-06\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  67%|██████▋   | 2/3 [04:08<02:04, 124.46s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      " group 0: name=predictor, lr=1e-06\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [06:12<00:00, 124.14s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      " group 0: name=predictor, lr=1e-06\n",
      " ---> [Stage A done] time: 372.61 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.15s/it, avg=3.79e+5, loss=2.6e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "379152.53125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.001, train_bs=8)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:45<00:00,  1.19s/it, avg=3.79e+5, loss=2.6e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "378881.53125\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 4490\n",
      "num predictor params = 0\n",
      "train_batch_size = 8\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.001, n_params=4490\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [02:02<04:04, 122.27s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [04:02<02:01, 121.08s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:01<00:00, 120.38s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n",
      " ---> [Stage B done] time: 361.14 sec\n",
      "\n",
      " === total train time: 886.23 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:45<00:00,  1.19s/it, avg=3.79e+5, loss=2.61e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "378922.09375\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPARATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 379849.500000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 379152.531250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 378881.531250\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 378922.125000\n",
      "============================================================\n",
      "\n",
      "Saved at: ../Result/NonParametric/KL/DFL_model_trained_separate_single_DRO.pkl\n",
      "\n",
      "==========================================================================================\n",
      "[Compare Scenario Filters - NonParametric]\n",
      "Predictor backbone : Stage-2 trained model\n",
      "Learned filter     : Stage-3 trained scenario_filter\n",
      "Methods            : ['learned', 'random', 'kmeans', 'kmedoids', 'hierarchical']\n",
      "Eval splits        : ('test',)\n",
      "==========================================================================================\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:44<00:00,  1.18s/it, avg=3.79e+5, loss=2.61e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][learned]\n",
      " test mean loss: 378922.093750\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.14s/it, avg=3.79e+5, loss=2.6e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][random]\n",
      " test mean loss: 379152.531250\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:59<00:00,  1.55s/it, avg=3.8e+5, loss=2.61e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmeans]\n",
      " test mean loss: 380154.593750\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:49<00:00,  1.29s/it, avg=3.82e+5, loss=2.62e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmedoids]\n",
      " test mean loss: 381649.000000\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.16s/it, avg=3.8e+5, loss=2.61e+5] "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][hierarchical]\n",
      " test mean loss: 380281.406250\n",
      "\n",
      "[Comparison Summary Rows]\n",
      "{'method': 'learned', 'test': 378922.09375}\n",
      "{'method': 'random', 'test': 379152.53125}\n",
      "{'method': 'kmeans', 'test': 380154.59375}\n",
      "{'method': 'kmedoids', 'test': 381649.0}\n",
      "{'method': 'hierarchical', 'test': 380281.40625}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode = \"single\"\n",
    "problem_mode = \"dro\"\n",
    "forecasting_mode = \"separate\"\n",
    "\n",
    "# 构建 manager 主要为了拿拓扑或者映射\n",
    "if optimization_mode == \"multi\":\n",
    "    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "else:\n",
    "    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == \"multi\":\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "# 第一次完整跑 Stage A + Stage B\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.eval_mode = \"discrete\"\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 1e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full = 200\n",
    "args.lambda_div = 1e5\n",
    "args.lambda_div_stage3 = 1e5\n",
    "# =========================================================\n",
    "# 1) 第一次：完整跑（Stage A + Stage B），并导出 stage2_artifact\n",
    "# =========================================================\n",
    "args.div_type = \"kl\"\n",
    "result_dro_single_kl = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(True, True, True, True, True),\n",
    "    stage2_artifact=None,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_single_kl,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/NonParametric/KL\"\n",
    ")\n",
    "\n",
    "compare_res_single_dro_kl = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_kl,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\", \"random\", \"kmeans\", \"kmedoids\", \"hierarchical\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_dro_kl)\n",
    "for row in summary_rows:\n",
    "    print(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fe5d28a",
   "metadata": {},
   "outputs": [],
   "source": [
    "compare_res_single_dro_all = merge_compare_results_learned_as_variants(\n",
    "    compare_res_single_dro_kl,\n",
    "    compare_res_single_dro_inner,\n",
    "    compare_res_single_dro_entropy,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a7ced60f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'backbone_predictor_source': \"stage2_artifact['dfl_after_stage2']\",\n",
       " 'learned_filter_source': \"base_result['dfl_trained'].scenario_filter\",\n",
       " 'method_names': ['learned', 'random', 'kmeans', 'kmedoids', 'hierarchical'],\n",
       " 'eval_splits': ('test',),\n",
       " 'details': {'learned': {'test_losses_compare_stage2backbone_learned': tensor([385460.4223, 325402.1198, 306543.3670, 300445.7518, 292113.9532,\n",
       "           275586.9559, 308774.4089, 306929.6793, 303151.5006, 300451.2780,\n",
       "           288091.4513, 294019.1633, 288319.9564, 340308.5941, 364978.8628,\n",
       "           390734.5043, 367116.7917, 366902.1367, 336103.0228, 313804.4996,\n",
       "           347740.3505, 269097.0505, 282580.8822, 270368.6207, 325672.6489,\n",
       "           350747.5011, 392596.9690, 322913.8684, 312234.6046, 308648.0697,\n",
       "           304487.0003, 280234.7144, 286139.3767, 336212.5513, 352826.6686,\n",
       "           353932.5999, 339897.3674, 344159.5921, 328471.7499, 333120.3380,\n",
       "           376043.1375, 374637.0088, 332686.6187, 300069.2593, 286704.8200,\n",
       "           280032.7316, 270084.3425, 272482.3769, 289878.0376, 267181.7394,\n",
       "           242826.9446, 245488.3582, 245588.0166, 322770.6194, 365632.7323,\n",
       "           304094.8391, 273352.0710, 244154.9527, 243499.0690, 236175.6251,\n",
       "           253249.3395, 267199.8960, 274071.0522, 249485.3046, 230206.7414,\n",
       "           243746.9055, 243089.3610, 245697.7663, 241915.5501, 272679.9699,\n",
       "           250792.2558, 238808.2664, 239886.0173, 234343.5852, 234437.0051,\n",
       "           243624.5334, 272007.9306, 256579.3458, 238475.5753, 262220.4807,\n",
       "           280706.2588, 255309.5243, 263928.2559, 262614.8413, 268758.9266,\n",
       "           268399.6427, 277485.2582, 275101.9897, 283730.6897, 289788.2460,\n",
       "           309170.1514, 338722.6208, 309372.8240, 306035.0395, 297242.3123,\n",
       "           268630.6897, 251381.7995, 256092.5389, 284376.3544, 292841.5410,\n",
       "           282617.3618, 271820.0694, 259018.3246, 269369.7632, 277012.8492,\n",
       "           259950.6904, 271068.7043, 256362.3456, 271618.5019, 292160.0233,\n",
       "           314024.5684, 319009.1900, 291159.7178, 276222.7667, 233840.9704,\n",
       "           244854.9723, 257255.7443, 281307.5418, 252859.1839, 243252.4772,\n",
       "           256865.8961, 277625.6855, 301974.5187, 299869.4664, 390686.3104,\n",
       "           475824.1126, 296595.3306, 251242.8433, 238575.4380, 244662.2726,\n",
       "           247408.2242, 264319.2883, 253890.5744, 258464.6094, 254675.1323,\n",
       "           324131.2629, 356888.0962, 410072.6246, 400781.2244, 476073.4868,\n",
       "           481364.1163, 395138.3798, 294531.7386, 266218.1825, 318227.0959,\n",
       "           369866.3256, 412838.2045, 446590.9596, 484258.4260, 569577.9421,\n",
       "           557440.8306, 628086.0488, 789709.8672, 801077.7097, 628561.9609,\n",
       "           539270.0343, 505446.9782, 419424.3449, 396824.5187, 498614.4524,\n",
       "           466233.0760, 516676.6225, 508113.9604, 445820.4790, 422391.6902,\n",
       "           450652.9384, 467939.2507, 509754.5074, 470530.1021, 463556.2534,\n",
       "           543602.1746, 592612.5331, 442903.8922, 429503.8002, 381161.8347,\n",
       "           324565.1973, 361541.7939, 375809.5553, 461115.0692, 529223.0126,\n",
       "           605808.4139, 596578.4897, 641776.7085, 462725.8273, 507491.2355,\n",
       "           531582.1584, 573348.0807, 610949.0305, 586680.6908, 588176.6731,\n",
       "           582017.3723, 585481.8467, 630503.6637, 589264.9752, 633382.2836,\n",
       "           645948.9520, 718520.9319, 604645.0079, 476486.3936, 442884.3265,\n",
       "           470146.9260, 557835.1608, 570457.8492, 595266.2736, 609087.3366,\n",
       "           618081.4715, 706735.3235, 836876.1522, 729124.8139, 649551.1865,\n",
       "           577494.6232, 596070.2046, 545766.6717, 536256.3923, 534284.5157,\n",
       "           507277.2259, 444183.6399, 425289.7411, 392967.1027, 396224.6132,\n",
       "           437220.0096, 457421.4142, 470689.5792, 507966.7175, 610378.4204,\n",
       "           590085.3761, 487080.4912, 532450.5018, 502878.5625, 492907.0337,\n",
       "           489653.3215, 481696.6469, 494812.0759, 533089.5064, 499980.3708,\n",
       "           500763.8130, 514638.0059, 523235.7287, 515797.5417, 384790.0861,\n",
       "           299476.6162, 309957.3428, 359531.9593, 454171.3544, 342418.5154,\n",
       "           321260.9349, 320297.4021, 422618.2399, 413332.3905, 367137.1826,\n",
       "           359723.5294, 350672.8547, 330228.8047, 313548.6315, 312148.6251,\n",
       "           308745.3686, 265939.2691, 285273.1384, 344256.5871, 356142.6709,\n",
       "           341392.5055, 368235.5430, 429900.4859, 455294.1646, 493084.8041,\n",
       "           498730.9724, 530491.3897, 487768.8400, 470702.9578, 524371.0080,\n",
       "           520463.1945, 486194.0111, 500686.3995, 479441.2900, 491526.8708,\n",
       "           545350.0778, 523780.1735, 458216.0065, 400303.0127, 305596.7615,\n",
       "           244029.9454, 235813.5851, 234721.5084, 233367.5808, 239081.6760,\n",
       "           246781.9470, 257116.2815, 240797.3137, 242767.0569, 253883.8347,\n",
       "           258420.3955, 249355.1517, 243704.3119, 249714.7752, 250351.0872,\n",
       "           245661.3542, 250972.7406, 270622.7030, 270161.9922, 261169.1390,\n",
       "           278511.1268, 251746.2124, 240367.9716], dtype=torch.float64)},\n",
       "  'random': {'test_losses_compare_stage2backbone_random': tensor([385865.9361, 327936.4339, 306877.6361, 300534.0313, 290601.0036,\n",
       "           277455.0608, 307573.3999, 306862.9493, 303414.5962, 301633.8058,\n",
       "           288145.5231, 294858.9834, 288722.5950, 342075.8743, 364287.0835,\n",
       "           392135.6958, 367121.7015, 371002.1990, 335477.2095, 312272.2869,\n",
       "           345184.7088, 271111.0683, 282618.2499, 271085.5065, 325658.0387,\n",
       "           350889.6787, 390707.9586, 322603.1306, 313544.0561, 309424.5686,\n",
       "           304694.8340, 280920.8344, 286398.1022, 335698.3034, 351935.6253,\n",
       "           353228.7899, 339202.0564, 343372.8349, 328246.6269, 335452.7531,\n",
       "           376249.1100, 374608.1549, 331601.1051, 299114.5267, 286471.0337,\n",
       "           279679.7309, 269335.1937, 273611.8773, 289920.3498, 270044.1141,\n",
       "           242966.8611, 246120.9859, 244824.8916, 322761.6964, 366171.6425,\n",
       "           303181.5675, 273643.4655, 244213.7122, 242963.9770, 236087.9130,\n",
       "           253454.7192, 266501.9164, 274186.9938, 250565.3739, 230180.3390,\n",
       "           244581.4949, 242014.6672, 245889.0547, 241867.2943, 271831.7334,\n",
       "           250854.4924, 239603.9857, 240428.4067, 234506.4257, 234249.5360,\n",
       "           244015.9901, 271842.6222, 256166.8350, 238961.0536, 263387.0560,\n",
       "           281097.1303, 256107.0205, 265064.8311, 262017.7291, 267923.8688,\n",
       "           267313.9368, 274586.9697, 276002.5310, 283770.6512, 290705.5588,\n",
       "           308455.1979, 338885.7410, 309023.0624, 305094.3568, 296697.5406,\n",
       "           269017.8652, 252667.7448, 256163.7580, 284361.7720, 292841.2017,\n",
       "           282238.5472, 271447.6782, 258986.0150, 268692.2762, 277061.3098,\n",
       "           260323.5601, 271066.1228, 256851.0735, 271928.2155, 290205.9668,\n",
       "           313294.7899, 319075.8702, 292160.2870, 275823.0659, 233664.3137,\n",
       "           245861.1697, 257156.8215, 280510.4219, 252183.0554, 242887.0026,\n",
       "           256829.4489, 278173.6221, 302900.1311, 301215.4019, 390082.5241,\n",
       "           476722.4296, 296197.1528, 250736.8774, 238897.3238, 246149.4483,\n",
       "           247271.9019, 264375.1210, 252933.9748, 258520.9531, 254672.6558,\n",
       "           323372.8916, 357034.5046, 413223.7489, 402550.1111, 477148.9598,\n",
       "           481742.0407, 397445.4874, 293308.6295, 265895.3821, 318470.4046,\n",
       "           369854.3767, 409758.6662, 447251.2110, 483338.2950, 569562.1046,\n",
       "           557054.6214, 628719.9308, 789657.7850, 799581.3242, 625861.5295,\n",
       "           538946.2856, 507423.2491, 427163.9259, 396756.7995, 498174.8764,\n",
       "           466585.2813, 517842.9132, 509670.9764, 445933.6176, 421946.6449,\n",
       "           449879.9261, 462010.6702, 508291.8530, 470647.3426, 463524.2410,\n",
       "           543582.8922, 594510.1708, 445014.5524, 426338.1350, 379086.7439,\n",
       "           325853.8890, 361827.4091, 375508.2592, 461083.7415, 529207.4064,\n",
       "           609470.5617, 597977.7109, 638334.2333, 461625.8403, 508756.8193,\n",
       "           531540.6953, 573309.2765, 615829.9739, 586205.7468, 590658.7736,\n",
       "           581192.3910, 587413.6362, 627417.6300, 594343.7084, 633179.0848,\n",
       "           642858.0038, 718635.7709, 606342.0135, 477132.7759, 442941.6074,\n",
       "           470162.2330, 559346.5116, 568522.9540, 595222.4484, 609027.3535,\n",
       "           619532.9988, 706784.2795, 837173.5286, 732100.6618, 651362.5649,\n",
       "           573633.8107, 596058.7692, 545341.2096, 537758.0773, 534201.5992,\n",
       "           508790.8321, 444504.4626, 429244.9762, 396038.1053, 396220.0505,\n",
       "           437225.6264, 457354.3936, 471432.4745, 507894.1630, 608679.4939,\n",
       "           588300.5845, 487048.4303, 531720.7377, 502827.5023, 493192.0802,\n",
       "           489692.4805, 482489.3747, 495113.3913, 533040.4200, 510535.5589,\n",
       "           502092.0834, 513939.2889, 526799.1514, 513684.8485, 388451.9168,\n",
       "           299647.8223, 309834.9211, 359627.3047, 452554.7032, 344389.7834,\n",
       "           321723.4352, 320165.9430, 422543.3610, 414155.5201, 367109.5134,\n",
       "           363215.9378, 350302.2863, 327902.6710, 313788.5449, 312614.2648,\n",
       "           310122.9287, 266318.6172, 286454.5050, 345651.1875, 357039.4563,\n",
       "           341092.2067, 368215.8539, 429915.3792, 457395.7070, 493096.8607,\n",
       "           498882.4022, 533098.3711, 488887.3573, 469702.4957, 524340.7461,\n",
       "           517291.7968, 488676.0535, 500217.3612, 479377.8648, 491467.0521,\n",
       "           545308.2135, 524781.6248, 457830.3076, 402523.6528, 308149.4888,\n",
       "           244320.6374, 236174.3434, 235717.0194, 233988.2708, 239077.9256,\n",
       "           246115.0522, 257007.3361, 242263.1911, 242850.2101, 253801.8630,\n",
       "           258724.0022, 249524.9185, 243624.8444, 249709.4631, 250467.4466,\n",
       "           246515.6412, 251199.2980, 271024.4736, 269621.0061, 260150.1460,\n",
       "           278853.6137, 250832.2680, 240498.0954], dtype=torch.float64)},\n",
       "  'kmeans': {'test_losses_compare_stage2backbone_kmeans': tensor([385662.7212, 327844.9252, 307019.9297, 302066.2398, 294092.1617,\n",
       "           276837.2056, 308110.1463, 306839.5375, 303793.6451, 301402.8809,\n",
       "           288391.7390, 294492.3106, 289146.9008, 340636.1608, 364482.3438,\n",
       "           391735.0085, 367784.5981, 371570.1573, 338299.1982, 316829.2377,\n",
       "           350860.0659, 271559.9275, 285599.3087, 270473.4052, 325403.3595,\n",
       "           351107.2240, 392719.3385, 324437.1136, 313370.1366, 308511.0819,\n",
       "           304903.6592, 282480.2779, 286103.4088, 337068.1709, 352714.6282,\n",
       "           355479.8239, 340138.2626, 343824.3493, 328591.4849, 335282.8734,\n",
       "           376336.5014, 374965.7047, 331834.4309, 300780.9489, 289354.4988,\n",
       "           281996.8124, 269918.1173, 273540.5324, 290202.4914, 270723.1517,\n",
       "           243066.7889, 245554.6167, 244730.8733, 322717.8325, 369746.5583,\n",
       "           305498.9087, 275321.2710, 244023.9965, 242995.9110, 236134.3698,\n",
       "           253353.4682, 266655.9353, 274751.2740, 251031.6409, 230873.3123,\n",
       "           244881.5825, 243519.0816, 246436.3490, 242154.6597, 272188.8974,\n",
       "           253527.8937, 239685.8403, 240794.9254, 234530.1603, 233951.2100,\n",
       "           244670.0919, 272000.2547, 257108.3415, 238296.6614, 265038.6109,\n",
       "           280190.6172, 256981.6550, 266285.1436, 263666.0510, 269555.7248,\n",
       "           269034.9607, 278019.9555, 278854.5025, 283666.6174, 289765.2957,\n",
       "           308548.9946, 338690.3056, 309291.0539, 304550.1207, 297289.3115,\n",
       "           270540.3275, 253030.4290, 256087.3477, 284330.3395, 292819.8401,\n",
       "           282965.3414, 271788.3543, 259135.9226, 269993.8232, 275799.2321,\n",
       "           260711.3421, 271052.1328, 257307.1760, 272132.1904, 292111.1558,\n",
       "           317520.3470, 320097.0221, 293094.8963, 276443.7303, 233630.3195,\n",
       "           246392.0214, 257522.6419, 280103.1493, 257075.0415, 243033.7057,\n",
       "           256928.9411, 279453.8838, 304236.9137, 302795.2133, 390961.0250,\n",
       "           481251.7140, 300478.3066, 252388.3925, 239854.7716, 247164.8612,\n",
       "           248826.5951, 264782.0938, 254847.5242, 260251.9516, 255363.4091,\n",
       "           323856.8826, 356665.3842, 413556.4384, 403964.5766, 481288.2526,\n",
       "           489875.6971, 397015.9679, 293216.5286, 266277.7970, 317386.5428,\n",
       "           369726.1814, 411386.9953, 448842.2607, 483474.0408, 569483.4301,\n",
       "           556940.9987, 633511.2595, 789977.8644, 804115.5772, 630621.9911,\n",
       "           543235.6880, 513022.9739, 426755.7229, 399151.1682, 499687.2254,\n",
       "           468026.4266, 516436.9117, 507992.1249, 448343.9274, 423175.9049,\n",
       "           451301.7514, 467163.5208, 507621.0135, 470159.4157, 463537.3276,\n",
       "           543550.1116, 596814.3397, 449068.6786, 427549.1505, 387194.5994,\n",
       "           325467.9375, 362425.3354, 374679.1482, 461027.2009, 529150.4300,\n",
       "           610882.0411, 598970.7131, 644009.1856, 464106.2941, 508464.9412,\n",
       "           531515.6078, 573246.0801, 614605.6118, 586609.4651, 590548.9586,\n",
       "           581703.0030, 587736.9519, 629699.1076, 593427.2582, 633111.6189,\n",
       "           647125.5818, 727704.6007, 610558.5208, 476465.6549, 442748.8437,\n",
       "           470380.9486, 557785.6389, 568773.9666, 595384.8329, 608898.9676,\n",
       "           618098.7274, 706703.1356, 849487.1731, 744064.1688, 649709.9853,\n",
       "           575744.5743, 595889.5653, 545218.8505, 536416.9929, 534780.3040,\n",
       "           508855.3937, 446593.6432, 427587.3605, 395483.2159, 396123.8797,\n",
       "           437128.3697, 457704.4948, 471357.9254, 508517.0867, 608833.7834,\n",
       "           589814.1849, 487003.9962, 532865.7668, 502743.2164, 492812.8166,\n",
       "           491420.0851, 484865.7884, 495237.5216, 533034.2806, 508508.3737,\n",
       "           508252.5402, 515123.2035, 523813.3769, 513802.1144, 391985.1964,\n",
       "           298750.5180, 308989.0410, 359734.7891, 453506.0764, 348105.9144,\n",
       "           322153.7411, 319875.1697, 422503.7331, 414259.8692, 367667.4542,\n",
       "           366087.6215, 350893.8152, 330179.5553, 312934.2869, 314150.3377,\n",
       "           312018.3761, 267157.1024, 284992.3664, 344826.6963, 358166.6559,\n",
       "           343993.1912, 368150.5007, 429907.5638, 458792.3143, 491836.9076,\n",
       "           499671.1084, 537332.4078, 499855.2248, 471907.3763, 524643.7739,\n",
       "           518219.6370, 487490.4521, 499781.6134, 479683.7348, 491408.9593,\n",
       "           545153.4550, 525922.1146, 457875.9337, 407843.6234, 310836.1225,\n",
       "           247060.9385, 236653.1313, 235619.8248, 234927.9317, 239144.8857,\n",
       "           247338.4699, 256701.4510, 243347.4314, 242770.8625, 255950.0400,\n",
       "           259851.0092, 252187.7326, 243688.8562, 250175.3005, 253326.7135,\n",
       "           246539.2884, 251901.3562, 271172.9401, 270362.0221, 263032.8193,\n",
       "           278794.7945, 253645.5549, 240361.8009], dtype=torch.float64)},\n",
       "  'kmedoids': {'test_losses_compare_stage2backbone_kmedoids': tensor([385478.5451, 330117.6086, 307390.0934, 304844.6258, 293269.8916,\n",
       "           278098.0624, 309428.6582, 306745.9548, 305235.7950, 302861.3178,\n",
       "           288526.3613, 296418.1077, 289445.3737, 340809.9918, 365222.1120,\n",
       "           391128.2989, 367984.9626, 377381.5501, 340760.7269, 322726.3623,\n",
       "           354344.9684, 272791.8722, 285735.5133, 271676.3435, 325411.6457,\n",
       "           351362.6128, 395114.4773, 327927.7171, 313457.2500, 308289.9099,\n",
       "           305245.7906, 282383.3034, 286079.9392, 337184.3291, 353467.9746,\n",
       "           357824.2366, 340239.1277, 343882.2557, 328734.4123, 336474.6335,\n",
       "           376127.9239, 374292.7438, 332445.5104, 305476.8195, 287970.3358,\n",
       "           283037.0043, 269779.3261, 274823.6974, 290725.9509, 271787.8102,\n",
       "           243050.7768, 247673.9716, 246418.5036, 322693.0251, 372892.9350,\n",
       "           306554.7936, 277793.3201, 244240.4396, 243180.6917, 237602.9028,\n",
       "           253293.4737, 268602.3444, 274967.3346, 251304.8476, 230590.2250,\n",
       "           245920.2747, 243135.7318, 246268.8911, 242932.8224, 272319.2428,\n",
       "           253232.9519, 240522.5857, 241474.8037, 234494.1417, 234425.1336,\n",
       "           245953.3327, 272496.7833, 257525.1904, 239509.0565, 264910.4154,\n",
       "           282958.0303, 257635.9665, 266275.8140, 266481.4609, 270270.6871,\n",
       "           268633.3885, 279603.3228, 280810.5667, 283693.1168, 291664.8385,\n",
       "           309063.3054, 338608.0310, 309451.1184, 305313.8577, 302352.0339,\n",
       "           273775.6661, 255247.6513, 256383.1575, 284295.8566, 292885.5717,\n",
       "           282695.8379, 272822.1272, 260787.3834, 271149.3991, 277079.0228,\n",
       "           262200.6964, 271009.1947, 258452.7901, 272122.1770, 292289.1868,\n",
       "           323666.5213, 320982.0386, 296792.5210, 278745.4911, 234310.0219,\n",
       "           246061.0457, 257477.0204, 281650.3000, 257476.3067, 242767.9407,\n",
       "           257123.3010, 279910.9385, 307400.9506, 304184.2847, 391336.1191,\n",
       "           482679.0110, 303507.1851, 254109.4487, 240447.1623, 247560.5205,\n",
       "           247699.3885, 265229.0364, 255387.5033, 260903.2528, 256067.1682,\n",
       "           325747.9456, 358317.4327, 416180.2734, 404139.3378, 482954.7760,\n",
       "           487898.3167, 406154.9480, 301009.6328, 266134.7053, 317651.2964,\n",
       "           370029.9492, 411988.2440, 450555.9662, 483993.0934, 569458.3956,\n",
       "           559613.8129, 638729.1333, 795723.6883, 814777.7916, 631613.0103,\n",
       "           545197.4281, 517618.0702, 427774.9881, 400237.9587, 504171.0497,\n",
       "           467795.2479, 522682.4992, 512404.5136, 451723.4255, 423229.4459,\n",
       "           451773.6039, 473982.9895, 514788.8600, 472840.3062, 463622.2627,\n",
       "           543506.3733, 601779.7429, 452335.1595, 430461.8978, 389512.5696,\n",
       "           325682.9143, 361904.2436, 377584.1223, 461003.3709, 529666.6847,\n",
       "           616342.1440, 601132.4091, 648891.2506, 468935.1795, 508436.2445,\n",
       "           531386.2938, 573204.0202, 624036.7983, 586375.8769, 592460.2940,\n",
       "           582269.9021, 586855.9358, 634410.0968, 596938.8966, 633269.4649,\n",
       "           651816.7585, 733071.4310, 616094.8251, 478942.2610, 443192.1956,\n",
       "           470925.6684, 560085.8921, 569446.4336, 595714.2466, 609087.3488,\n",
       "           618332.1851, 706573.8114, 851212.1501, 748536.4023, 650284.0893,\n",
       "           577401.4253, 595812.5061, 546089.8661, 538657.3993, 534964.5952,\n",
       "           509345.7028, 448412.0752, 430410.6252, 397232.0615, 396027.3436,\n",
       "           437075.8526, 458175.9582, 473557.1238, 508913.4712, 615882.8709,\n",
       "           592158.8444, 486958.6142, 533363.5766, 502656.1626, 493113.9312,\n",
       "           491019.7207, 485846.0680, 497961.9889, 532857.8139, 511970.0296,\n",
       "           509643.7207, 515248.3095, 531187.2973, 523313.2145, 399634.7782,\n",
       "           300183.1355, 311051.4740, 360915.2662, 460176.7378, 353207.4075,\n",
       "           322231.1613, 320669.4452, 422444.4383, 417646.9961, 368228.5329,\n",
       "           366354.6707, 352117.8638, 334397.6278, 314795.6039, 315176.6935,\n",
       "           314709.1844, 268264.9243, 286702.4889, 346019.2435, 364661.2683,\n",
       "           345293.1485, 368137.5618, 429848.4665, 457795.7243, 497982.6838,\n",
       "           502322.3275, 539993.7668, 506076.9306, 472700.5936, 524571.3821,\n",
       "           521741.1856, 491594.2882, 503327.1772, 479561.6909, 491375.3869,\n",
       "           545069.6588, 524866.5223, 458371.3245, 408650.7986, 314731.4810,\n",
       "           247558.8388, 237746.4887, 235593.6898, 235405.3729, 239576.3832,\n",
       "           247118.3861, 257490.1642, 243143.6922, 243369.4486, 255440.1903,\n",
       "           259930.7644, 254138.2773, 244219.7749, 249970.5407, 254059.4039,\n",
       "           246522.4911, 252732.6727, 271683.1404, 271225.5038, 263227.5969,\n",
       "           279075.3302, 253361.5269, 240539.7248], dtype=torch.float64)},\n",
       "  'hierarchical': {'test_losses_compare_stage2backbone_hierarchical': tensor([386413.4463, 328828.6115, 307606.0832, 301417.6492, 292627.1930,\n",
       "           277721.3812, 308878.6106, 306819.0514, 304695.0139, 301045.7602,\n",
       "           288564.1354, 296491.6183, 289141.8079, 341956.4954, 363747.5071,\n",
       "           391728.3105, 367555.9042, 370853.4206, 339936.2727, 316151.6942,\n",
       "           352365.5822, 270778.2866, 283771.1884, 270485.5776, 325410.8111,\n",
       "           350712.7414, 391497.0869, 324765.4645, 313404.1157, 308722.5287,\n",
       "           304443.9519, 282479.8161, 286024.7535, 336654.3415, 352708.5831,\n",
       "           356156.6678, 340296.4211, 343905.6490, 328718.1249, 336218.0372,\n",
       "           376175.3566, 373978.9188, 331951.5123, 302199.0151, 287997.9685,\n",
       "           281379.7276, 270287.6836, 273974.6691, 290631.2756, 270174.4781,\n",
       "           243425.0397, 245933.6529, 245556.5892, 322697.3615, 369494.6623,\n",
       "           303607.0429, 275739.3779, 244303.5025, 243744.7713, 236040.9709,\n",
       "           253137.9320, 266979.6802, 274361.3067, 251062.5206, 231026.0953,\n",
       "           246296.6458, 242974.9919, 245922.6741, 242527.0871, 271777.5656,\n",
       "           252846.2123, 240126.4511, 241526.4902, 234383.8507, 234649.7876,\n",
       "           245075.3904, 272251.9588, 256589.1532, 239011.8979, 263671.7639,\n",
       "           281973.7060, 256842.7242, 266427.4572, 262853.5902, 270950.5008,\n",
       "           268809.8783, 279241.6616, 277791.6700, 283711.7796, 290647.3462,\n",
       "           308411.9471, 338678.3352, 309322.5873, 304917.4914, 298679.6870,\n",
       "           272521.2868, 254202.7868, 256145.1520, 284322.9864, 293102.4801,\n",
       "           282337.9761, 272117.3472, 260218.2424, 270246.9488, 276527.5839,\n",
       "           261038.1306, 271034.6660, 257252.9508, 271729.9423, 290713.9039,\n",
       "           314797.0587, 320364.7952, 294530.6229, 276877.4065, 233514.2505,\n",
       "           245313.1561, 257486.1741, 279132.2262, 256481.9459, 243478.8421,\n",
       "           256915.6162, 279328.9518, 307167.4576, 302857.4148, 392656.0880,\n",
       "           477794.3497, 299590.6219, 252156.8670, 239627.3927, 245747.1769,\n",
       "           247777.0500, 264352.9391, 254158.9494, 259403.1210, 255322.9029,\n",
       "           325554.6306, 355774.8717, 415118.0148, 403310.2816, 480697.9824,\n",
       "           486538.2176, 402844.5295, 297410.7161, 265900.7250, 317884.8031,\n",
       "           369969.0272, 410456.4581, 449657.7356, 483639.7852, 569472.7151,\n",
       "           557121.3469, 632867.2210, 788475.4018, 803717.1145, 627562.4024,\n",
       "           542885.8957, 516981.8857, 425643.9001, 400232.7737, 500330.8259,\n",
       "           467849.1016, 520646.0380, 508462.8110, 446993.4476, 422902.5043,\n",
       "           450565.2606, 465735.6739, 512610.4429, 471418.7908, 463500.8834,\n",
       "           543531.4312, 597263.2419, 447805.6252, 430147.1461, 387285.5655,\n",
       "           324850.3313, 362691.3704, 374028.7684, 461067.8321, 529738.6135,\n",
       "           613844.8074, 598992.8651, 640558.3267, 463199.1995, 510101.6647,\n",
       "           531497.4533, 573280.8826, 619264.8407, 586411.5638, 590203.4728,\n",
       "           580421.1714, 587275.7956, 630102.3206, 593195.9635, 633124.3196,\n",
       "           646742.4695, 728101.3184, 609227.5908, 477285.4880, 442724.2993,\n",
       "           471000.3030, 557785.4465, 568749.6354, 595240.0987, 608906.8587,\n",
       "           618002.3979, 706683.2933, 847704.6704, 738043.0361, 650090.0994,\n",
       "           574522.0874, 595878.7010, 546316.4344, 537016.6093, 534607.9460,\n",
       "           506948.0288, 444182.5969, 428574.4799, 393997.2266, 396082.8489,\n",
       "           437124.4843, 457281.9648, 472277.1936, 508436.0774, 604149.4882,\n",
       "           589014.2143, 486983.8245, 533112.5736, 502735.9137, 493121.1403,\n",
       "           490663.4414, 484789.0637, 495057.5415, 532962.0114, 511934.4427,\n",
       "           508221.3455, 514875.4071, 531226.7438, 520403.9709, 394439.4588,\n",
       "           299653.0387, 310279.6341, 359912.1425, 455778.6954, 349560.6750,\n",
       "           322768.2453, 320222.0661, 422457.7101, 414986.1596, 368097.1207,\n",
       "           366638.5984, 350018.5840, 331922.2395, 314850.9330, 313796.6044,\n",
       "           313373.7999, 267867.3857, 285781.4169, 345533.2690, 361119.1882,\n",
       "           344518.9009, 368160.1111, 429973.6257, 456341.6238, 493928.1787,\n",
       "           500676.9579, 533961.1802, 499051.5877, 473436.8930, 524296.1649,\n",
       "           520546.3019, 489048.6163, 501791.3895, 480419.9960, 491408.6665,\n",
       "           545224.4549, 524707.0734, 458431.8358, 403788.0879, 310162.0385,\n",
       "           246528.5657, 237187.0958, 235728.6899, 235444.9488, 239629.4965,\n",
       "           246816.5137, 257102.5363, 242854.3021, 243409.9759, 255070.0877,\n",
       "           258915.3861, 250920.4343, 243945.6849, 249806.6206, 252919.1020,\n",
       "           246203.2868, 252363.9402, 271083.2101, 270844.9502, 260577.6312,\n",
       "           278649.9122, 253539.1944, 240589.6183], dtype=torch.float64)}},\n",
       " 'summary_mean': {'learned': {'test': 378922.09375},\n",
       "  'random': {'test': 379152.53125},\n",
       "  'kmeans': {'test': 380154.59375},\n",
       "  'kmedoids': {'test': 381649.0},\n",
       "  'hierarchical': {'test': 380281.40625}}}"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "compare_res_single_dro_kl"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "eeeff46c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "args.train_batch_size = 8\n",
      "args.test_batch_size = 8\n",
      "run_stage2: True run_stage3: True reused_stage2: False multi: False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:06<00:00,  5.47it/s, avg=4.26e+5, loss=2.69e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425584.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:53<00:00,  1.41s/it, avg=3.8e+5, loss=2.61e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "379849.5\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-05, train_bs=8)\n",
      "train_mode = dfl\n",
      "scenario_filter type = RandomScenarioSelector\n",
      "has_learnable_filter = False\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 0\n",
      "num predictor params = 401489\n",
      "train_batch_size = 8\n",
      "===== optimizer param groups =====\n",
      "group 0: name=predictor, lr=1e-05, n_params=401489\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  33%|███▎      | 1/3 [02:02<04:04, 122.39s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      " group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  67%|██████▋   | 2/3 [04:03<02:01, 121.92s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      " group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [06:04<00:00, 121.58s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      " group 0: name=predictor, lr=1e-05\n",
      " ---> [Stage A done] time: 364.91 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:42<00:00,  1.12s/it, avg=3.77e+5, loss=2.59e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "377374.125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.01, train_bs=8)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.15s/it, avg=3.77e+5, loss=2.59e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "377090.875\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 4490\n",
      "num predictor params = 0\n",
      "train_batch_size = 8\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.01, n_params=4490\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [02:00<04:00, 120.39s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      " group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [04:02<02:01, 121.16s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      " group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:03<00:00, 121.04s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      " group 0: name=filter, lr=0.01\n",
      " ---> [Stage B done] time: 363.12 sec\n",
      "\n",
      " === total train time: 876.68 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.14s/it, avg=3.77e+5, loss=2.59e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "377098.40625\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPARATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 379849.500000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 377374.125000\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 377090.843750\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 377098.406250\n",
      "============================================================\n",
      "\n",
      "Saved at: ../Result/NonParametric/KL/DFL_model_trained_separate_single_DRO.pkl\n",
      "\n",
      "==========================================================================================\n",
      "[Compare Scenario Filters - NonParametric]\n",
      "Predictor backbone : Stage-2 trained model\n",
      "Learned filter     : Stage-3 trained scenario_filter\n",
      "Methods            : ['learned', 'random', 'kmeans', 'kmedoids', 'hierarchical']\n",
      "Eval splits        : ('test',)\n",
      "==========================================================================================\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.14s/it, avg=3.77e+5, loss=2.59e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][learned]\n",
      " test mean loss: 377098.406250\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:43<00:00,  1.14s/it, avg=3.77e+5, loss=2.59e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][random]\n",
      " test mean loss: 377374.125000\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:56<00:00,  1.50s/it, avg=3.78e+5, loss=2.6e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmeans]\n",
      " test mean loss: 377898.656250\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:46<00:00,  1.22s/it, avg=3.79e+5, loss=2.6e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmedoids]\n",
      " test mean loss: 378690.656250\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:44<00:00,  1.17s/it, avg=3.78e+5, loss=2.6e+5] "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][hierarchical]\n",
      " test mean loss: 377898.125000\n",
      "\n",
      "[Comparison Summary Rows]\n",
      "{'method': 'learned', 'test': 377098.40625}\n",
      "{'method': 'random', 'test': 377374.125}\n",
      "{'method': 'kmeans', 'test': 377898.65625}\n",
      "{'method': 'kmedoids', 'test': 378690.65625}\n",
      "{'method': 'hierarchical', 'test': 377898.125}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode = \"single\"\n",
    "problem_mode = \"dro\"\n",
    "forecasting_mode = \"separate\"\n",
    "\n",
    "# 构建 manager 主要为了拿拓扑或者映射\n",
    "if optimization_mode == \"multi\":\n",
    "    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "else:\n",
    "    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == \"multi\":\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "# 第一次完整跑 Stage A + Stage B\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.eval_mode = \"discrete\"\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-5\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full = 200\n",
    "args.lambda_div = 1e5\n",
    "args.lambda_div_stage3 = 1e5\n",
    "# =========================================================\n",
    "# 1) 第一次：完整跑（Stage A + Stage B），并导出 stage2_artifact\n",
    "# =========================================================\n",
    "args.div_type = \"kl\"\n",
    "result_dro_single_kl = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(True, True, True, True, True),\n",
    "    stage2_artifact=None,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_single_kl,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/NonParametric/KL\"\n",
    ")\n",
    "\n",
    "compare_res_single_dro_kl = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_kl,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\", \"random\", \"kmeans\", \"kmedoids\", \"hierarchical\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_dro_kl)\n",
    "for row in summary_rows:\n",
    "    print(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9917dc61",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "args.train_batch_size = 8\n",
      "args.test_batch_size = 8\n",
      "run_stage2: True run_stage3: True reused_stage2: True multi: False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "\n",
      " ---> Reusing passed stage2_artifact, skip Stage A and jump to Stage B\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.001, train_bs=8)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 4490\n",
      "num predictor params = 0\n",
      "train_batch_size = 8\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.001, n_params=4490\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [02:05<04:11, 125.98s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [04:06<02:02, 122.71s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:08<00:00, 122.88s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n",
      " ---> [Stage B done] time: 368.65 sec\n",
      "\n",
      " === total train time: 369.06 sec ===\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPARATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 379849.500000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 379152.531250\n",
      "============================================================\n",
      "\n",
      "Saved at: ../Result/NonParametric/Entropy/DFL_model_trained_separate_single_DRO.pkl\n",
      "\n",
      "==========================================================================================\n",
      "[Compare Scenario Filters - NonParametric]\n",
      "Predictor backbone : Stage-2 trained model\n",
      "Learned filter     : Stage-3 trained scenario_filter\n",
      "Methods            : ['learned']\n",
      "Eval splits        : ('test',)\n",
      "==========================================================================================\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:45<00:00,  1.20s/it, avg=3.79e+5, loss=2.6e+5] \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][learned]\n",
      " test mean loss: 378870.718750\n",
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "args.train_batch_size = 8\n",
      "args.test_batch_size = 8\n",
      "run_stage2: True run_stage3: True reused_stage2: True multi: False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "\n",
      " ---> Reusing passed stage2_artifact, skip Stage A and jump to Stage B\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.001, train_bs=8)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 4490\n",
      "num predictor params = 0\n",
      "train_batch_size = 8\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.001, n_params=4490\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [02:02<04:05, 122.61s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [04:04<02:02, 122.25s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:05<00:00, 121.86s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      " group 0: name=filter, lr=0.001\n",
      " ---> [Stage B done] time: 365.60 sec\n",
      "\n",
      " === total train time: 365.98 sec ===\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPARATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 379849.500000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 379152.531250\n",
      "============================================================\n",
      "\n",
      "Saved at: ../Result/NonParametric/Inner/DFL_model_trained_separate_single_DRO.pkl\n",
      "\n",
      "==========================================================================================\n",
      "[Compare Scenario Filters - NonParametric]\n",
      "Predictor backbone : Stage-2 trained model\n",
      "Learned filter     : Stage-3 trained scenario_filter\n",
      "Methods            : ['learned']\n",
      "Eval splits        : ('test',)\n",
      "==========================================================================================\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 38/38 [00:44<00:00,  1.17s/it, avg=3.79e+5, loss=2.6e+5] "
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][learned]\n",
      " test mean loss: 378838.625000\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'merge_compare_results_learned_as_variants' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[0;32mIn [10]\u001b[0m, in \u001b[0;36m<cell line: 72>\u001b[0;34m()\u001b[0m\n\u001b[1;32m     54\u001b[0m save_run_result(result_dro_single_inner, forecasting_mode, out_dir\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m../Result/NonParametric/Inner\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     56\u001b[0m compare_res_single_dro_inner \u001b[38;5;241m=\u001b[39m compare_scenario_filters_with_stage3_learned_non_parametric(\n\u001b[1;32m     57\u001b[0m     base_result\u001b[38;5;241m=\u001b[39mresult_dro_single_inner,\n\u001b[1;32m     58\u001b[0m     args\u001b[38;5;241m=\u001b[39margs,\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m     68\u001b[0m     verbose\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m,\n\u001b[1;32m     69\u001b[0m )\n\u001b[0;32m---> 72\u001b[0m compare_res_single_dro_all \u001b[38;5;241m=\u001b[39m \u001b[43mmerge_compare_results_learned_as_variants\u001b[49m(\n\u001b[1;32m     73\u001b[0m     compare_res_single_dro_kl,\n\u001b[1;32m     74\u001b[0m     compare_res_single_dro_inner,\n\u001b[1;32m     75\u001b[0m     compare_res_single_dro_entropy,\n\u001b[1;32m     76\u001b[0m )\n\u001b[1;32m     78\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m[Comparison Summary Rows]\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     79\u001b[0m summary_rows \u001b[38;5;241m=\u001b[39m summarize_compare_result(compare_res_single_dro_kl)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'merge_compare_results_learned_as_variants' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "shared_stage2_artifact = result_dro_single_kl[\"stage2_artifact\"]\n",
    "args.div_type = 'entropy'\n",
    "result_dro_single_entropy = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(False, False, False, False, False),\n",
    "    stage2_artifact=shared_stage2_artifact, \n",
    ")\n",
    "\n",
    "save_run_result(result_dro_single_entropy, forecasting_mode, out_dir=\"../Result/NonParametric/Entropy\")\n",
    "\n",
    "compare_res_single_dro_entropy = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_entropy,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "\n",
    "args.lambda_div = 5e8\n",
    "args.lambda_div_stage3 = 5e8\n",
    "args.div_type = 'inner'\n",
    "result_dro_single_inner = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(False, False, False, False, False),\n",
    "    stage2_artifact=shared_stage2_artifact,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(result_dro_single_inner, forecasting_mode, out_dir=\"../Result/NonParametric/Inner\")\n",
    "\n",
    "compare_res_single_dro_inner = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_inner,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "\n",
    "compare_res_single_dro_all = merge_compare_results_learned_as_variants(\n",
    "    compare_res_single_dro_kl,\n",
    "    compare_res_single_dro_inner,\n",
    "    compare_res_single_dro_entropy,\n",
    ")\n",
    "\n",
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_dro_kl)\n",
    "for row in summary_rows:\n",
    "    print(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "85ed709c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Comparison Summary Rows]\n",
      "{'method': 'random', 'test': 379152.53125}\n",
      "{'method': 'kmeans', 'test': 380154.59375}\n",
      "{'method': 'kmedoids', 'test': 381649.0}\n",
      "{'method': 'hierarchical', 'test': 380281.40625}\n",
      "{'method': 'learned (KL)', 'test': 378922.09375}\n",
      "{'method': 'learned (Inner)', 'test': 378838.625}\n",
      "{'method': 'learned (Entropy)', 'test': 378870.71875}\n"
     ]
    }
   ],
   "source": [
    "\n",
    "compare_res_single_dro_all = merge_compare_results_learned_as_variants(\n",
    "    compare_res_single_dro_kl,\n",
    "    compare_res_single_dro_inner,\n",
    "    compare_res_single_dro_entropy,\n",
    ")\n",
    "\n",
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_dro_all)\n",
    "for row in summary_rows:\n",
    "    print(row)\n",
    "\n",
    "with open('../Result/Non_parametric/compare_res_single_dro_all.pkl', 'wb') as f:\n",
    "    pickle.dump(compare_res_single_dro_all, f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "96725b57",
   "metadata": {},
   "outputs": [],
   "source": [
    "import copy\n",
    "\n",
    "def merge_compare_results_learned_as_variants(compare_res_single_dro_kl,\n",
    "                                             compare_res_single_dro_inner,\n",
    "                                             compare_res_single_dro_entropy,\n",
    "                                             learned_key=\"learned\",\n",
    "                                             labels=(\"KL\", \"Inner\", \"Entropy\"),\n",
    "                                             keep_baselines_from=\"kl\"):\n",
    "    \"\"\"\n",
    "    合并三份 compare_result，输出结构与原来一致，只是把 learned 拆成：\n",
    "      learned (KL), learned (Inner), learned (Entropy)\n",
    "    其它 random/kmeans/... 只保留一份（默认取 KL 那份）。\n",
    "\n",
    "    返回：merged_compare_result (dict)\n",
    "    \"\"\"\n",
    "    def rename_learned(res, new_learned_name):\n",
    "        res = copy.deepcopy(res)\n",
    "\n",
    "        # method_names\n",
    "        res[\"method_names\"] = [new_learned_name if m == learned_key else m\n",
    "                               for m in res.get(\"method_names\", [])]\n",
    "\n",
    "        # summary_mean\n",
    "        if \"summary_mean\" in res and learned_key in res[\"summary_mean\"]:\n",
    "            res[\"summary_mean\"][new_learned_name] = res[\"summary_mean\"].pop(learned_key)\n",
    "\n",
    "        # details：把 learned 这一项整体挪到新名字，同时把内部 key 的后缀也替换掉\n",
    "        if \"details\" in res and learned_key in res[\"details\"]:\n",
    "            old_block = res[\"details\"].pop(learned_key)\n",
    "            new_block = {}\n",
    "            for k, v in old_block.items():\n",
    "                # k 形如 test_losses_compare_stage2backbone_learned\n",
    "                new_k = k.replace(f\"_{learned_key}\", f\"_{new_learned_name}\")\n",
    "                new_block[new_k] = v\n",
    "            res[\"details\"][new_learned_name] = new_block\n",
    "\n",
    "        return res\n",
    "\n",
    "    r_kl   = rename_learned(compare_res_single_dro_kl,      f\"{learned_key} ({labels[0]})\")\n",
    "    r_in   = rename_learned(compare_res_single_dro_inner,   f\"{learned_key} ({labels[1]})\")\n",
    "    r_ent  = rename_learned(compare_res_single_dro_entropy, f\"{learned_key} ({labels[2]})\")\n",
    "\n",
    "    base = {\"kl\": r_kl, \"inner\": r_in, \"entropy\": r_ent}[keep_baselines_from]\n",
    "    merged = copy.deepcopy(base)\n",
    "\n",
    "    # baselines：非 learned(...) 的方法名（只从 base 取一份）\n",
    "    baselines = [m for m in merged.get(\"method_names\", [])\n",
    "                 if not (isinstance(m, str) and m.startswith(f\"{learned_key} (\"))]\n",
    "\n",
    "    # 清掉 base 里 learned(...)，后面按顺序重新加 3 个 learned(...)\n",
    "    for m in list(merged.get(\"method_names\", [])):\n",
    "        if isinstance(m, str) and m.startswith(f\"{learned_key} (\"):\n",
    "            merged[\"method_names\"].remove(m)\n",
    "\n",
    "    for key in list(merged.get(\"details\", {}).keys()):\n",
    "        if isinstance(key, str) and key.startswith(f\"{learned_key} (\"):\n",
    "            merged[\"details\"].pop(key)\n",
    "\n",
    "    for key in list(merged.get(\"summary_mean\", {}).keys()):\n",
    "        if isinstance(key, str) and key.startswith(f\"{learned_key} (\"):\n",
    "            merged[\"summary_mean\"].pop(key)\n",
    "\n",
    "    # 追加三份 learned(...)\n",
    "    for rr in (r_kl, r_in, r_ent):\n",
    "        learned_names = [m for m in rr.get(\"method_names\", [])\n",
    "                         if isinstance(m, str) and m.startswith(f\"{learned_key} (\")]\n",
    "        if len(learned_names) != 1:\n",
    "            raise ValueError(f\"Cannot uniquely find learned variant in one result: {learned_names}\")\n",
    "        ln = learned_names[0]\n",
    "\n",
    "        merged.setdefault(\"details\", {})[ln] = rr.get(\"details\", {}).get(ln, {})\n",
    "        merged.setdefault(\"summary_mean\", {})[ln] = rr.get(\"summary_mean\", {}).get(ln, {})\n",
    "        merged.setdefault(\"method_names\", []).append(ln)\n",
    "\n",
    "    # 最终 method_names 顺序：learned(KL/Inner/Entropy) + baselines\n",
    "    merged[\"method_names\"] = [f\"{learned_key} ({labels[0]})\",\n",
    "                              f\"{learned_key} ({labels[1]})\",\n",
    "                              f\"{learned_key} ({labels[2]})\"] + baselines\n",
    "\n",
    "    return merged"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "64a9e8a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "run_stage2: True run_stage3: True reused_stage2: False multi: False\n",
      "args.batch_size None\n",
      "args.dfl_batch_size 32\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.75it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425584.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:26<00:00,  2.61s/it, avg=3.88e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387580.6875\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-05, bs=32)\n",
      "train_mode = dfl\n",
      "scenario_filter type = RandomScenarioSelector\n",
      "has_learnable_filter = False\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 0\n",
      "num predictor params = 401489\n",
      "===== optimizer param groups =====\n",
      "group 0: name=predictor, lr=1e-05, n_params=401489\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  33%|███▎      | 1/3 [01:08<02:17, 68.73s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  67%|██████▋   | 2/3 [02:17<01:08, 68.75s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:26<00:00, 68.93s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n",
      " ---> [Stage A done] time: 207.08 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.06s/it, avg=3.8e+5, loss=2.35e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "379673.78125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.01, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.08s/it, avg=3.77e+5, loss=2.34e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "377452.71875\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 4490\n",
      "num predictor params = 0\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.01, n_params=4490\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [01:08<02:16, 68.44s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [02:16<01:08, 68.20s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [03:24<00:00, 68.27s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n",
      " ---> [Stage B done] time: 204.80 sec\n",
      "\n",
      " === total train time: 486.60 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.78e+5, loss=2.34e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "377500.5\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPARATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387580.687500\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 379673.781250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 377452.687500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 377500.500000\n",
      "============================================================\n",
      "\n",
      "Saved at: ../Result/NonParametric/KL/DFL_model_trained_separate_single_SO.pkl\n",
      "\n",
      "==========================================================================================\n",
      "[Compare Scenario Filters - NonParametric]\n",
      "Predictor backbone : Stage-2 trained model\n",
      "Learned filter     : Stage-3 trained scenario_filter\n",
      "Methods            : ['learned', 'random', 'kmeans', 'kmedoids', 'hierarchical']\n",
      "Eval splits        : ('test',)\n",
      "==========================================================================================\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.17s/it, avg=3.78e+5, loss=2.34e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][learned]\n",
      " test mean loss: 377500.500000\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.08s/it, avg=3.8e+5, loss=2.35e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][random]\n",
      " test mean loss: 379673.781250\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:32<00:00,  3.29s/it, avg=3.82e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmeans]\n",
      " test mean loss: 382488.781250\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.18s/it, avg=3.89e+5, loss=2.39e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmedoids]\n",
      " test mean loss: 388730.875000\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.83e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][hierarchical]\n",
      " test mean loss: 383380.812500\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode = \"single\"\n",
    "problem_mode = \"so\"\n",
    "forecasting_mode = \"separate\"\n",
    "\n",
    "# 构建 manager 主要为了拿拓扑或者映射\n",
    "if optimization_mode == \"multi\":\n",
    "    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "else:\n",
    "    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == \"multi\":\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "# 第一次完整跑 Stage A + Stage B\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.eval_mode = \"discrete\"\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-5\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full = 200\n",
    "args.lambda_div = 1e3\n",
    "\n",
    "# =========================================================\n",
    "# 1) 第一次：完整跑（Stage A + Stage B），并导出 stage2_artifact\n",
    "# =========================================================\n",
    "args.div_type = \"kl\"\n",
    "result_so_single_kl = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(True, True, True, True, True),\n",
    "    stage2_artifact=None,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_so_single_kl,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/NonParametric/KL\"\n",
    ")\n",
    "\n",
    "compare_res_single_so_kl = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_so_single_kl,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\", \"random\", \"kmeans\", \"kmedoids\", \"hierarchical\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "654c180c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Comparison Summary Rows]\n"
     ]
    },
    {
     "ename": "NameError",
     "evalue": "name 'summarize_compare_result' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Input \u001b[0;32mIn [1]\u001b[0m, in \u001b[0;36m<cell line: 2>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m[Comparison Summary Rows]\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m----> 2\u001b[0m summary_rows \u001b[38;5;241m=\u001b[39m \u001b[43msummarize_compare_result\u001b[49m(compare_res_single_so_kl)\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m row \u001b[38;5;129;01min\u001b[39;00m summary_rows:\n\u001b[1;32m      4\u001b[0m     \u001b[38;5;28mprint\u001b[39m(row)\n",
      "\u001b[0;31mNameError\u001b[0m: name 'summarize_compare_result' is not defined"
     ]
    }
   ],
   "source": [
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_so_kl)\n",
    "for row in summary_rows:\n",
    "    print(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a7f25ece",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "run_stage2: True run_stage3: True reused_stage2: False multi: False\n",
      "args.batch_size None\n",
      "args.dfl_batch_size 32\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.67it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425584.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.44s/it, avg=3.8e+5, loss=2.56e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "380123.3125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-05, bs=32)\n",
      "train_mode = dfl\n",
      "scenario_filter type = RandomScenarioSelector\n",
      "has_learnable_filter = False\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 0\n",
      "num predictor params = 401489\n",
      "===== optimizer param groups =====\n",
      "group 0: name=predictor, lr=1e-05, n_params=401489\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  33%|███▎      | 1/3 [02:00<04:00, 120.09s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  67%|██████▋   | 2/3 [04:02<02:01, 121.68s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [06:05<00:00, 121.88s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n",
      " ---> [Stage A done] time: 365.66 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.48s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "378132.8125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.01, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 1.0\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.49s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "377634.625\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 4490\n",
      "num predictor params = 0\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.01, n_params=4490\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [02:06<04:13, 126.96s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [04:13<02:06, 126.80s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:16<00:00, 125.64s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n",
      " ---> [Stage B done] time: 376.92 sec\n",
      "\n",
      " === total train time: 884.50 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.50s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "377662.375\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPARATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 380123.312500\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 378132.812500\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 377634.625000\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 377662.375000\n",
      "============================================================\n",
      "\n",
      "Saved at: ../Result/NonParametric/KL/DFL_model_trained_separate_single_DRO.pkl\n",
      "\n",
      "==========================================================================================\n",
      "[Compare Scenario Filters - NonParametric]\n",
      "Predictor backbone : Stage-2 trained model\n",
      "Learned filter     : Stage-3 trained scenario_filter\n",
      "Methods            : ['learned', 'random', 'kmeans', 'kmedoids', 'hierarchical']\n",
      "Eval splits        : ('test',)\n",
      "==========================================================================================\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.53s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][learned]\n",
      " test mean loss: 377662.375000\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.49s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][random]\n",
      " test mean loss: 378132.812500\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:56<00:00,  5.60s/it, avg=3.79e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmeans]\n",
      " test mean loss: 378761.937500\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.52s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][kmedoids]\n",
      " test mean loss: 379957.468750\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.54s/it, avg=3.79e+5, loss=2.54e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Compare][hierarchical]\n",
      " test mean loss: 378854.656250\n",
      "\n",
      "[Comparison Summary Rows]\n",
      "{'method': 'learned', 'test': 377662.375}\n",
      "{'method': 'random', 'test': 378132.8125}\n",
      "{'method': 'kmeans', 'test': 378761.9375}\n",
      "{'method': 'kmedoids', 'test': 379957.46875}\n",
      "{'method': 'hierarchical', 'test': 378854.65625}\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode = \"single\"\n",
    "problem_mode = \"dro\"\n",
    "forecasting_mode = \"separate\"\n",
    "\n",
    "# 构建 manager 主要为了拿拓扑或者映射\n",
    "if optimization_mode == \"multi\":\n",
    "    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "else:\n",
    "    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == \"multi\":\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "# 第一次完整跑 Stage A + Stage B\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.eval_mode = \"discrete\"\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-5\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full = 200\n",
    "args.lambda_div = 1e3\n",
    "\n",
    "# =========================================================\n",
    "# 1) 第一次：完整跑（Stage A + Stage B），并导出 stage2_artifact\n",
    "# =========================================================\n",
    "args.div_type = \"kl\"\n",
    "result_dro_single_kl = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(True, True, True, True, True),\n",
    "    stage2_artifact=None,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_single_kl,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/NonParametric/KL\"\n",
    ")\n",
    "\n",
    "compare_res_single_dro_kl = compare_scenario_filters_with_stage3_learned_non_parametric(\n",
    "    base_result=result_dro_single_kl,\n",
    "    args=args,\n",
    "    problem_mode=problem_mode,\n",
    "    optimization_mode=optimization_mode,\n",
    "    quantiles=quantiles,\n",
    "    models_s=models_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    eval_splits=(\"test\",),\n",
    "    method_names=[\"learned\", \"random\", \"kmeans\", \"kmedoids\", \"hierarchical\"],\n",
    "    seed=0,\n",
    "    verbose=True,\n",
    ")\n",
    "\n",
    "print(\"\\n[Comparison Summary Rows]\")\n",
    "summary_rows = summarize_compare_result(compare_res_single_dro_kl)\n",
    "for row in summary_rows:\n",
    "    print(row)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "125ecf48",
   "metadata": {},
   "outputs": [],
   "source": [
    "optimization_mode = 'multi'\n",
    "problem_mode = 'dro'\n",
    "forecasting_mode = 'seperate'\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.lr_decay = 1\n",
    "args.filter_lr_decay = 1\n",
    "args.dfl_lr_decay = 1\n",
    "\n",
    "args.eval_mode = \"discrete\"\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-5\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full = 200\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "\n",
    "\n",
    "args.div_type = 'kl'\n",
    "args.lambda_div = 1e3            # 你也可以改成 1e3\n",
    "args.lambda_div_stage3 = 1e3\n",
    "\n",
    "result_dro_multi_kl = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(True, True, True, True, True),\n",
    "    stage2_artifact=None,   # 第一次从头跑\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_multi_kl,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/Non_parametric/KL\"\n",
    ")\n",
    "\n",
    "shared_stage2_artifact = result_dro_multi_kl[\"stage2_artifact\"]\n",
    "\n",
    "args.div_type = 'entropy'\n",
    "args.lambda_div = 1e3\n",
    "args.lambda_div_stage3 = 1e3\n",
    "\n",
    "result_dro_multi_entropy = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(False, False, False, True, True),\n",
    "    stage2_artifact=shared_stage2_artifact,\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_multi_entropy,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/Non_parametric/Entropy\"\n",
    ")\n",
    "\n",
    "args.div_type = 'inner'\n",
    "args.lambda_div = 1e3\n",
    "args.lambda_div_stage3 = 1e3\n",
    "\n",
    "result_dro_multi_inner = run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    eval_flags=(False, False, False, True, True),\n",
    "    stage2_artifact=shared_stage2_artifact,\n",
    ")\n",
    "\n",
    "save_run_result(\n",
    "    result_dro_multi_inner,\n",
    "    forecasting_mode,\n",
    "    out_dir=\"../Result/Non_parametric/Inner\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "231ab7f8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "697253a3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 15\n",
      "K_model 5\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 0.1\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:06<00:00,  1.57it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425584.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:55<00:00,  5.54s/it, avg=3.8e+5, loss=2.56e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "380123.3125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-05, bs=32)\n",
      "train_mode = dfl\n",
      "scenario_filter type = RandomScenarioSelector\n",
      "has_learnable_filter = False\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 0\n",
      "num predictor params = 401489\n",
      "===== optimizer param groups =====\n",
      "group 0: name=predictor, lr=1e-05, n_params=401489\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  33%|███▎      | 1/3 [02:05<04:10, 125.49s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl):  67%|██████▋   | 2/3 [04:11<02:06, 126.09s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [06:16<00:00, 125.57s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      "  group 0: name=predictor, lr=1e-05\n",
      " ---> [Stage A done] time: 376.90 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.51s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "378119.5625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.01, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 0.1\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.60s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "377919.6875\n",
      "train_mode = filter_only\n",
      "scenario_filter type = ScenarioFilter\n",
      "has_learnable_filter = True\n",
      "lr_decay = 1.0 filter_lr_decay = 1.0 dfl_lr_decay = 1.0\n",
      "num filter params = 3845\n",
      "num predictor params = 0\n",
      "===== optimizer param groups =====\n",
      "group 0: name=filter, lr=0.01, n_params=3845\n",
      "==================================\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  33%|███▎      | 1/3 [02:07<04:14, 127.20s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 1] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only):  67%|██████▋   | 2/3 [04:10<02:05, 125.19s/it]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 2] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:15<00:00, 125.32s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[Epoch 3] lr after decay:\n",
      "  group 0: name=filter, lr=0.01\n",
      " ---> [Stage B done] time: 375.98 sec\n",
      "\n",
      " === total train time: 844.00 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.58s/it, avg=3.78e+5, loss=2.54e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "377884.125\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425584.031250\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 380123.312500\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 378119.562500\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 377919.687500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 377884.093750\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_DRO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_DRO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='dro'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.lr_decay= 1\n",
    "args.filter_lr_decay= 1\n",
    "args.dfl_lr_decay= 1\n",
    "args.dfl_epochs = 3\n",
    "args.dfl_lr = 1e-5\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result_dro=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result_dro,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b6c8a5cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 0.1\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.42s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "378119.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:43<00:00,  4.39s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "377919.6875\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=3, lr=0.01, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:10<00:00, 123.58s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 370.74 sec\n",
      "\n",
      " === total train time: 370.74 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:43<00:00,  4.36s/it, avg=3.78e+5, loss=2.54e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "377894.5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "args.lr_decay= 1\n",
    "args.filter_lr_decay= 3\n",
    "args.dfl_lr_decay= 1\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 3\n",
    "args.filter_lr = 1e-2\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode=\"dro\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dfdef2b4",
   "metadata": {},
   "outputs": [],
   "source": [
    "args.div_type='kl'\n",
    "args.lr_decay= 1\n",
    "args.filter_lr_decay= 3\n",
    "args.dfl_lr_decay= 1\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 3\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode=\"dro\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "55200d5e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1.0 1.0 0.5\n"
     ]
    }
   ],
   "source": [
    "\n",
    "lr_decay = float(getattr(args, \"lr_decay\", 1.0))\n",
    "filter_lr_decay = float(getattr(args, \"filter_lr_decay\", lr_decay))\n",
    "dfl_lr_decay = float(getattr(args, \"dfl_lr_decay\", lr_decay))\n",
    "print(lr_decay,filter_lr_decay,dfl_lr_decay)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ea968dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "eps_uniform 0.1\n",
      "tau_gumbel 0.1\n",
      "eval_mode discrete\n",
      "avoid_rand_duplicate False\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:46<00:00,  4.69s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "378119.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.45s/it, avg=3.78e+5, loss=2.54e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "377919.6875\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=3, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:42<00:00, 134.11s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 402.35 sec\n",
      "\n",
      " === total train time: 402.35 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:55<00:00,  5.55s/it, avg=3.78e+5, loss=2.54e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "377895.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "args.lr_decay= 1\n",
    "args.filter_lr_decay= 3\n",
    "args.dfl_lr_decay= 0.5\n",
    "#args.lambda_div=1e3 inner\n",
    "\n",
    "\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 3\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode=\"dro\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bde7f2ba",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.83it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.55s/it, avg=3.8e+5, loss=2.56e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "380113.15625\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [06:00<00:00, 120.04s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 360.13 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.57s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "379848.5625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.58s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "379813.375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [10:06<00:00, 121.23s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 606.14 sec\n",
      "\n",
      " === total train time: 1057.83 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.58s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379824.28125\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 380113.187500\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 379848.562500\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 379813.375000\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 379824.312500\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_DRO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_DRO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='dro'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-5\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result_dro=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result_dro,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "8c2905e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def DFL_train(\n",
    "    dfl,\n",
    "    train_dataset,\n",
    "    args,\n",
    "    problem_mode=\"saa\",\n",
    "    train_mode=\"dfl\",\n",
    "    filter_kwargs=None,\n",
    "    lambda_div=1e5,\n",
    "):\n",
    "    print('new version')\n",
    "    dfl.train()\n",
    "    device = next(dfl.parameters()).device\n",
    "\n",
    "    # ---- 同步 scenario_filter 的 eval 配置 ----\n",
    "    eval_mode = str(getattr(args, \"eval_mode\", \"soft\")).lower()\n",
    "    avoid_rand_duplicate = bool(getattr(args, \"avoid_rand_duplicate\", False))\n",
    "    if getattr(dfl, \"scenario_filter\", None) is not None:\n",
    "        if hasattr(dfl.scenario_filter, \"eval_mode\"):\n",
    "            dfl.scenario_filter.eval_mode = eval_mode\n",
    "        if hasattr(dfl.scenario_filter, \"avoid_rand_duplicate\"):\n",
    "            dfl.scenario_filter.avoid_rand_duplicate = avoid_rand_duplicate\n",
    "\n",
    "    if filter_kwargs is None:\n",
    "        filter_kwargs = {\n",
    "            \"tau_gumbel\": getattr(args, \"tau_mix\", 1.0),\n",
    "            \"eps_uniform\": getattr(args, \"eps_uniform\", 0.1),\n",
    "        }\n",
    "\n",
    "    div_type = str(getattr(args, \"div_type\", \"inner\")).lower()\n",
    "    div_eps = float(getattr(args, \"div_eps\", 1e-8))\n",
    "\n",
    "    if str(problem_mode).lower() in [\"saa\", \"so\"]:\n",
    "        Lmin = Lmax = eps_value = None\n",
    "    else:\n",
    "        Lmin = args.Lmin\n",
    "        Lmax = args.Lmax\n",
    "        eps_value = args.eps_value\n",
    "\n",
    "    if train_mode == \"filter_only\" and getattr(dfl, \"scenario_filter\", None) is not None:\n",
    "        for p in dfl.parameters():\n",
    "            p.requires_grad = False\n",
    "        for p in dfl.scenario_filter.parameters():\n",
    "            p.requires_grad = True\n",
    "\n",
    "        optim_params_filter = [p for p in dfl.scenario_filter.parameters() if p.requires_grad]\n",
    "        optim_params_predictor = []\n",
    "\n",
    "        filter_lr = float(getattr(args, \"lr\", getattr(args, \"filter_lr\", 1e-3)))\n",
    "        optim = torch.optim.Adam(\n",
    "            optim_params_filter,\n",
    "            lr=filter_lr,\n",
    "            # weight_decay=float(getattr(args, \"weight_decay\", 0.0)),\n",
    "        )\n",
    "\n",
    "    else:\n",
    "        for p in dfl.parameters():\n",
    "            p.requires_grad = True\n",
    "\n",
    "        optim_params_filter = (\n",
    "            [p for p in dfl.scenario_filter.parameters() if p.requires_grad]\n",
    "            if getattr(dfl, \"scenario_filter\", None) is not None else []\n",
    "        )\n",
    "        optim_params_predictor = [p for p in dfl.predictor.parameters() if p.requires_grad]\n",
    "\n",
    "        filter_lr = float(getattr(args, \"filter_lr\", 1e-3))\n",
    "        predictor_lr = float(getattr(args, \"dfl_lr\", getattr(args, \"lr\", 1e-5)))\n",
    "\n",
    "        optim = torch.optim.Adam(\n",
    "            [\n",
    "                {\"params\": optim_params_filter, \"lr\": filter_lr},\n",
    "                {\"params\": optim_params_predictor, \"lr\": predictor_lr},\n",
    "            ],\n",
    "            weight_decay=float(getattr(args, \"weight_decay\", 0.0)),\n",
    "        )\n",
    "\n",
    "    # ===== 学习率衰减参数 =====\n",
    "    lr_decay = float(getattr(args, \"lr_decay\", 1.0))   # 例如 0.95\n",
    "    min_lr = float(getattr(args, \"min_lr\", 0.0))       # 可选，防止 lr 太小\n",
    "\n",
    "    loader = DataLoader(\n",
    "        train_dataset,\n",
    "        batch_size=int(getattr(args, \"dfl_train_batch_size\", getattr(args, \"dfl_batch_size\", getattr(args, \"batch_size\", 8)))),\n",
    "        shuffle=True,\n",
    "        drop_last=False,\n",
    "    )\n",
    "\n",
    "    epochs = int(getattr(args, \"epochs\", 1))\n",
    "    train_logs = []\n",
    "\n",
    "    for ep in tqdm(range(epochs), desc=f\"Train ({train_mode})\", leave=True):\n",
    "        epoch_task_loss, epoch_div_loss, samples_cnt = 0.0, 0.0, 0\n",
    "        pbar = tqdm(loader, desc=f\"Ep {ep+1}/{epochs}\", leave=False)\n",
    "\n",
    "        for X_all, y_true_real in pbar:\n",
    "            X_all = X_all.to(device).float()\n",
    "            y_true_real = y_true_real.to(device).float()\n",
    "            optim.zero_grad(set_to_none=True)\n",
    "\n",
    "            kwargs = dict(\n",
    "                solver=getattr(args, \"solver\", None),\n",
    "                return_aux=False,\n",
    "                return_filter_aux=True,\n",
    "                predictor_n_samples=int(getattr(args, \"S_full\", getattr(args, \"N_scen\", 50))),\n",
    "                filter_kwargs=filter_kwargs,\n",
    "            )\n",
    "\n",
    "            if str(problem_mode).lower() == \"dro\":\n",
    "                dtype = getattr(dfl, \"optnet_dtype\", torch.float64)\n",
    "                kwargs.update(\n",
    "                    hourly_load_min_sys=torch.as_tensor(Lmin, device=device, dtype=dtype),\n",
    "                    hourly_load_max_sys=torch.as_tensor(Lmax, device=device, dtype=dtype),\n",
    "                    eps=torch.as_tensor(eps_value, device=device, dtype=dtype),\n",
    "                )\n",
    "\n",
    "            out = dfl(X_all, y_true_real, **kwargs)\n",
    "            if isinstance(out, (tuple, list)):\n",
    "                loss_vec = out[0]\n",
    "                aux_filter = out[-1]\n",
    "            else:\n",
    "                loss_vec = out\n",
    "                aux_filter = None\n",
    "\n",
    "            task_loss_val = loss_vec.mean()\n",
    "            div_loss_val = torch.tensor(0.0, device=device)\n",
    "\n",
    "            if aux_filter is not None and \"p\" in aux_filter and lambda_div > 0:\n",
    "                p = aux_filter[\"p\"].clamp_min(div_eps)\n",
    "                p = p / p.sum(dim=-1, keepdim=True).clamp_min(div_eps)\n",
    "                Bc, Kc, _ = p.shape\n",
    "\n",
    "                if div_type == \"inner\" and Kc > 1:\n",
    "                    M = torch.bmm(p, p.transpose(1, 2))\n",
    "                    eye = torch.eye(Kc, device=device, dtype=torch.bool).unsqueeze(0).expand(Bc, -1, -1)\n",
    "                    div_loss_val = (M[~eye] ** 2).mean()\n",
    "\n",
    "                elif div_type == \"kl\" and Kc > 1:\n",
    "                    pi, pj = p.unsqueeze(2), p.unsqueeze(1)\n",
    "                    kl_ij = (pi * (pi.log() - pj.log())).sum(dim=-1)\n",
    "                    kl_ji = (pj * (pj.log() - pi.log())).sum(dim=-1)\n",
    "                    skl = 0.5 * (kl_ij + kl_ji)\n",
    "                    eye = torch.eye(Kc, device=device, dtype=torch.bool).unsqueeze(0).expand(Bc, -1, -1)\n",
    "                    div_loss_val = -skl[~eye].mean()\n",
    "\n",
    "                elif div_type == \"entropy\":\n",
    "                    H = -(p * p.log()).sum(dim=-1).mean()\n",
    "                    div_loss_val = -H\n",
    "\n",
    "            loss = task_loss_val + float(lambda_div) * div_loss_val\n",
    "            loss.backward()\n",
    "\n",
    "            if len(optim_params_filter) > 0:\n",
    "                torch.nn.utils.clip_grad_norm_(optim_params_filter, 1.0)\n",
    "            if len(optim_params_predictor) > 0:\n",
    "                torch.nn.utils.clip_grad_norm_(optim_params_predictor, 1.0)\n",
    "\n",
    "            optim.step()\n",
    "\n",
    "            Bsz = X_all.shape[0]\n",
    "            epoch_task_loss += float(task_loss_val.detach().cpu()) * Bsz\n",
    "            epoch_div_loss += float(div_loss_val.detach().cpu()) * Bsz\n",
    "            samples_cnt += Bsz\n",
    "\n",
    "            current_lrs = [pg[\"lr\"] for pg in optim.param_groups]\n",
    "            pbar.set_postfix(\n",
    "                Task=float(task_loss_val.detach().cpu()),\n",
    "                Div=float(div_loss_val.detach().cpu()),\n",
    "                Type=div_type,\n",
    "                LR=current_lrs,\n",
    "            )\n",
    "\n",
    "        # ===== 每个 epoch 结束后做 lr 衰减 =====\n",
    "        for param_group in optim.param_groups:\n",
    "            new_lr = max(param_group[\"lr\"] * lr_decay, min_lr)\n",
    "            param_group[\"lr\"] = new_lr\n",
    "\n",
    "        train_logs.append({\n",
    "            \"task\": epoch_task_loss / max(samples_cnt, 1),\n",
    "            \"div\": epoch_div_loss / max(samples_cnt, 1),\n",
    "            \"div_type\": div_type,\n",
    "            \"lr\": [pg[\"lr\"] for pg in optim.param_groups],\n",
    "        })\n",
    "\n",
    "    return dfl, train_logs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "1b619205",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.51s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "379848.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:43<00:00,  4.40s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "379816.5625\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [10:15<00:00, 123.17s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 615.85 sec\n",
      "\n",
      " === total train time: 615.85 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.58s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379832.96875\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 5\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode='dro',          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "28ac3119",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.58s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "379848.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.41s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "379813.375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=3, lr=0.01, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:01<00:00, 120.55s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 361.66 sec\n",
      "\n",
      " === total train time: 361.66 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.43s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379730.625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "args.lr_decay=0.5\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 3\n",
    "args.filter_lr = 1e-2\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode='dro',          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "71ae5073",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:46<00:00,  4.60s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "379848.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.45s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "379813.375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=3, lr=0.01, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [06:12<00:00, 124.32s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 372.98 sec\n",
      "\n",
      " === total train time: 372.98 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:47<00:00,  4.72s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379774.9375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "args.lr_decay=0.8\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 3\n",
    "args.filter_lr = 1e-2\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode='dro',          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "4fc571b6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.47s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "379848.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.45s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "379813.375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.01, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [02:01<00:00, 121.64s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 121.65 sec\n",
      "\n",
      " === total train time: 121.65 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.48s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379724.34375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 1\n",
    "args.filter_lr = 1e-2\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode='dro',          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "86156d57",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:46<00:00,  4.70s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "379848.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:46<00:00,  4.61s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "379813.375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [02:01<00:00, 121.62s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 121.62 sec\n",
      "\n",
      " === total train time: 121.62 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.53s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379772.40625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 5\n",
    "args.filter_lr = 1e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode='dro',          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83a210d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.55s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "379848.5625\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.53s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "379813.375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [10:06<00:00, 121.26s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 606.32 sec\n",
      "\n",
      " === total train time: 606.32 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.43s/it, avg=3.8e+5, loss=2.55e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "379810.125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 5\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result_dro,\n",
    "    args=args,\n",
    "    problem_mode='dro',          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1f1c58d2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.87it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.42s/it, avg=3.88e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387594.8125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=1, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 1/1 [01:06<00:00, 66.41s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 66.60 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.87e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "386950.21875\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=1, lr=0.005, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.08s/it, avg=3.87e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "387113.84375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:07<00:00, 67.97s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 67.98 sec\n",
      "\n",
      " === total train time: 176.53 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.12s/it, avg=3.87e+5, loss=2.38e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "387085.3125\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387594.875000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 386950.218750\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 387113.812500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 387085.343750\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 1    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 1\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "aee9b2b2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:04<00:00,  2.15it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.88e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387594.8125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:18<00:00, 66.09s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 198.29 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385932.21875\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "386083.40625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:41<00:00, 68.30s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 341.49 sec\n",
      "\n",
      " === total train time: 581.96 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.20s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386057.09375\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387594.875000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385932.250000\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 386083.437500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 386057.093750\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 5\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91a5b522",
   "metadata": {},
   "outputs": [],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 1\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e4 #entropy\n",
    "args.lambda_div_stage3=1e6\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "4695ede1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.19s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385932.21875\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.16s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386083.40625\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "new version\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:41<00:00, 68.39s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 341.95 sec\n",
      "\n",
      " === total train time: 341.95 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.21s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385883.3125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 5\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e3 #entropy\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9dc39cf9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.22s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385932.21875\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.17s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "385876.59375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:08<00:00, 68.41s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 68.41 sec\n",
      "\n",
      " === total train time: 68.41 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385857.40625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 1\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e4 #entropy\n",
    "args.lambda_div_stage3=1e6\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5adfa507",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.69it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.17s/it, avg=3.88e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387594.8125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:21<00:00, 67.29s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 201.89 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.13s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385932.21875\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.10s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "386083.40625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:42<00:00, 68.49s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 342.44 sec\n",
      "\n",
      " === total train time: 586.63 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.23s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386057.09375\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387594.875000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385932.250000\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 386083.437500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 386057.093750\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='dro'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 1    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 1\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result_dro=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result_dro,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "abd880d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:04<00:00,  2.11it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.52s/it, avg=3.87e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387081.78125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:19<00:00, 66.48s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 199.63 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.10s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385416.65625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "385901.09375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:47<00:00, 69.45s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 347.25 sec\n",
      "\n",
      " === total train time: 588.98 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385826.0\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387081.781250\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385416.656250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 385901.093750\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 385826.031250\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 1    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 1\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "66d684bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Final stage (stage3_after):\n",
      "385826.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "test_result = run_DFL_non_parametric_test(\n",
    "    args=args,\n",
    "    dfl_trained=result['dfl_trained'],\n",
    "    problem_mode=problem_mode,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6504579e",
   "metadata": {},
   "outputs": [],
   "source": [
    "load_old_ckpt_into_scenario_filter(\n",
    "    scenario_filter=new_result['dfl_trained'].scenario_filter,\n",
    "    ckpt_path=\"checkpoints/nufilter_non_parametric.pt\",\n",
    "    T=24,\n",
    "    hidden=128,\n",
    "    strict=True,\n",
    "    device=device,\n",
    ")\n",
    "\n",
    "test_result = run_DFL_non_parametric_test(\n",
    "    args=args,\n",
    "    dfl_trained=new_result['dfl_trained'],\n",
    "    problem_mode=problem_mode,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "22bd1e5f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_init.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.460801\n",
      "0.bias               shape=(128,) norm=1.282785\n",
      "2.weight             shape=(5, 128) norm=1.303147\n",
      "2.bias               shape=(5,) norm=0.096323\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.25s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Final stage (stage3_after):\n",
      "385550.75\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "orig_state = copy.deepcopy(result['dfl_trained'].scenario_filter.state_dict())\n",
    "\n",
    "load_old_ckpt_into_scenario_filter(\n",
    "    scenario_filter=result['dfl_trained'].scenario_filter,\n",
    "    ckpt_path=\"checkpoints/nufilter_init.pt\",\n",
    "    T=24,\n",
    "    hidden=128,\n",
    "    strict=True,\n",
    "    device=device,\n",
    ")\n",
    "# 3) 测试\n",
    "test_result_old = run_DFL_non_parametric_test(\n",
    "    args=args,\n",
    "    dfl_trained=result['dfl_trained'],\n",
    "    problem_mode=problem_mode,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "# 4) 恢复原 filter 参数\n",
    "result['dfl_trained'].scenario_filter.load_state_dict(orig_state, strict=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "8c7bbdcd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_regular.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.491444\n",
      "0.bias               shape=(128,) norm=1.281435\n",
      "2.weight             shape=(5, 128) norm=1.340244\n",
      "2.bias               shape=(5,) norm=0.096176\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.22s/it, avg=3.85e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Final stage (stage3_after):\n",
      "385279.40625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<All keys matched successfully>"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "orig_state = copy.deepcopy(result['dfl_trained'].scenario_filter.state_dict())\n",
    "\n",
    "load_old_ckpt_into_scenario_filter(\n",
    "    scenario_filter=result['dfl_trained'].scenario_filter,\n",
    "    ckpt_path=\"checkpoints/nufilter_regular.pt\",\n",
    "    T=24,\n",
    "    hidden=128,\n",
    "    strict=True,\n",
    "    device=device,\n",
    ")\n",
    "# 3) 测试\n",
    "test_result_old = run_DFL_non_parametric_test(\n",
    "    args=args,\n",
    "    dfl_trained=result['dfl_trained'],\n",
    "    problem_mode=problem_mode,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "# 4) 恢复原 filter 参数\n",
    "result['dfl_trained'].scenario_filter.load_state_dict(orig_state, strict=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "2c4054c7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test:  10%|█         | 1/10 [00:02<00:25,  2.81s/it, avg=3.14e+5, loss=3.14e+5]/home/zyz/miniconda3/envs/Meta_DFL/lib/python3.8/site-packages/diffcp/cone_program.py:371: UserWarning: Solved/Inaccurate.\n",
      "  warnings.warn(\"Solved/Inaccurate.\")\n",
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.18s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Final stage (stage3_after):\n",
      "385826.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "test_result = run_DFL_non_parametric_test(\n",
    "    args=args,\n",
    "    dfl_trained=result['dfl_trained'],\n",
    "    problem_mode=problem_mode,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "dc8f4350",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_result=copy.deepcopy(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bf61608b",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_result['dfl_trained'].scenario_filter"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "950ed22e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DFL_model_nonparametric_SingleNode(\n",
       "  (optnet_DA): SingleNode_Reserve_SAA_DA_OptNet(\n",
       "    (layer): CvxpyLayer()\n",
       "  )\n",
       "  (optnet_RT): SingleNode_Reserve_RT_OptNet(\n",
       "    (layer): CvxpyLayer()\n",
       "  )\n",
       "  (predictor): Multi_nonparametric_quantile_predictor(\n",
       "    (models): ModuleDict(\n",
       "      (4-2-0): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-1): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-2): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-3): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-4): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-5): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-6): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-7): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-8): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-9): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "      (4-2-10): ANN_quantiles_non_parametric(\n",
       "        (net): Sequential(\n",
       "          (0): Linear(in_features=7, out_features=128, bias=True)\n",
       "          (1): ReLU()\n",
       "          (2): Dropout(p=0.0, inplace=False)\n",
       "          (3): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (4): ReLU()\n",
       "          (5): Dropout(p=0.0, inplace=False)\n",
       "          (6): Linear(in_features=128, out_features=128, bias=True)\n",
       "          (7): ReLU()\n",
       "          (8): Dropout(p=0.0, inplace=False)\n",
       "          (9): Linear(in_features=128, out_features=19, bias=True)\n",
       "        )\n",
       "      )\n",
       "    )\n",
       "  )\n",
       "  (scenario_filter): ScenarioFilter(\n",
       "    (net): Sequential(\n",
       "      (0): Linear(in_features=24, out_features=128, bias=True)\n",
       "      (1): ReLU()\n",
       "      (2): Linear(in_features=128, out_features=5, bias=True)\n",
       "    )\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_result['dfl_trained']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "47634fe7",
   "metadata": {},
   "outputs": [],
   "source": [
    "nufilter_init"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f1b7930a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_non_parametric.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.500168\n",
      "0.bias               shape=(128,) norm=1.285351\n",
      "2.weight             shape=(5, 128) norm=1.344288\n",
      "2.bias               shape=(5,) norm=0.094388\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "ScenarioFilter(\n",
       "  (net): Sequential(\n",
       "    (0): Linear(in_features=24, out_features=128, bias=True)\n",
       "    (1): ReLU()\n",
       "    (2): Linear(in_features=128, out_features=5, bias=True)\n",
       "  )\n",
       ")"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "load_old_ckpt_into_scenario_filter(\n",
    "    scenario_filter=new_result['dfl_trained'].scenario_filter,\n",
    "    ckpt_path=\"checkpoints/nufilter_non_parametric.pt\",\n",
    "    T=24,\n",
    "    hidden=128,\n",
    "    strict=True,\n",
    "    device=device,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5cbc04cb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.82it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.21s/it, avg=3.87e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387081.78125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:27<00:00, 69.27s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 207.81 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.34s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385416.65625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.26s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "385901.09375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [06:14<00:00, 74.98s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 374.89 sec\n",
      "\n",
      " === total train time: 628.68 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.41s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385826.0\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387081.781250\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385416.656250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 385901.093750\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 385826.031250\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "args.eval_mode = \"discrete\"\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 5\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79d7baa0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.73it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:27<00:00,  2.77s/it, avg=3.87e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387081.78125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:46<00:00, 75.38s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 226.14 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.20s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385416.65625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.22s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "386208.5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:46<00:00, 69.28s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 346.42 sec\n",
      "\n",
      " === total train time: 616.79 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.20s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386279.71875\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387081.781250\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385416.656250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 386208.531250\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 386279.718750\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "args.eval_mode = \"discrete\"\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 5\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a316878a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.88it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.31s/it, avg=3.88e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387594.8125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:19<00:00, 66.55s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 199.67 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385932.21875\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.88e+5, loss=2.4e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "388158.5625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:34<00:00, 66.87s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 334.38 sec\n",
      "\n",
      " === total train time: 576.89 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.16s/it, avg=3.88e+5, loss=2.4e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "387879.40625\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387594.875000\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385932.250000\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 388158.562500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 387879.375000\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 5\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "702366b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:04<00:00,  2.16it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:44<00:00,  4.45s/it, avg=3.8e+5, loss=2.56e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "380113.15625\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [06:04<00:00, 121.48s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 364.44 sec\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.53s/it, avg=3.8e+5, loss=2.55e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "379848.5625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:45<00:00,  4.60s/it, avg=3.8e+5, loss=2.56e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "380436.875\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [09:54<00:00, 118.81s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 594.05 sec\n",
      "\n",
      " === total train time: 1049.83 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:46<00:00,  4.62s/it, avg=3.8e+5, loss=2.56e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "380443.0\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | DRO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 380113.187500\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 379848.562500\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 380436.937500\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 380443.031250\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_DRO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_DRO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='dro'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 5\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 1\n",
    "args.eps_uniform = 0.1\n",
    "result_dro=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result_dro,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "8ccff5ff",
   "metadata": {},
   "outputs": [],
   "source": [
    "import copy\n",
    "import torch\n",
    "\n",
    "# ============================================================\n",
    "# 1) 旧 NuFilter 定义（只用于加载 checkpoint 结构）\n",
    "# ============================================================\n",
    "class NuFilter(torch.nn.Module):\n",
    "    def __init__(self, T: int = 24, hidden: int = 128, K_model: int = 10):\n",
    "        super().__init__()\n",
    "        self.K_model = K_model\n",
    "        self.net = torch.nn.Sequential(\n",
    "            torch.nn.Linear(T, hidden),\n",
    "            torch.nn.ReLU(),\n",
    "            torch.nn.Linear(hidden, K_model),\n",
    "        )\n",
    "\n",
    "    def forward(self, Y_full_BST: torch.Tensor, tau: float = 1.0):\n",
    "        logits = self.net(Y_full_BST)          # (B,S,K_model)\n",
    "        logits = logits.transpose(1, 2)        # (B,K_model,S)\n",
    "        w = torch.softmax(logits / max(float(tau), 1e-6), dim=-1)\n",
    "        return w\n",
    "\n",
    "# ============================================================\n",
    "# 2) 从旧 ckpt 加载并复制到新 ScenarioFilter.net\n",
    "# ============================================================\n",
    "def load_old_ckpt_into_scenario_filter(\n",
    "    scenario_filter,\n",
    "    ckpt_path=\"checkpoints/nufilter_regular.pt\",\n",
    "    T=24,\n",
    "    hidden=128,\n",
    "    strict=True,\n",
    "    device=None,\n",
    "):\n",
    "    \"\"\"\n",
    "    将旧版 NuFilter checkpoint 参数复制到新版 ScenarioFilter.net\n",
    "    \"\"\"\n",
    "    if device is None:\n",
    "        device = next(scenario_filter.parameters()).device\n",
    "\n",
    "    K_model = int(scenario_filter.K_model)\n",
    "    if K_model <= 0:\n",
    "        raise ValueError(f\"ScenarioFilter.K_model must be > 0, got {K_model}\")\n",
    "\n",
    "    # 建一个旧版 NuFilter 骨架\n",
    "    old_model = NuFilter(T=T, hidden=hidden, K_model=K_model).to(device)\n",
    "\n",
    "    # 读取旧 checkpoint\n",
    "    ckpt = torch.load(ckpt_path, map_location=device)\n",
    "\n",
    "    # 兼容两种保存格式：\n",
    "    # 1) 直接是 state_dict\n",
    "    # 2) {\"state_dict\": ...}\n",
    "    if isinstance(ckpt, dict) and \"state_dict\" in ckpt:\n",
    "        ckpt = ckpt[\"state_dict\"]\n",
    "\n",
    "    # 加载到旧模型\n",
    "    msg_old = old_model.load_state_dict(ckpt, strict=strict)\n",
    "    print(\"\\n[Load old NuFilter ckpt]\")\n",
    "    print(\"ckpt_path =\", ckpt_path)\n",
    "    print(\"old model load result =\", msg_old)\n",
    "\n",
    "    # 再把 old_model.net 的参数复制到新 scenario_filter.net\n",
    "    msg_new = scenario_filter.net.load_state_dict(old_model.net.state_dict(), strict=strict)\n",
    "    print(\"\\n[Copy old NuFilter.net -> new ScenarioFilter.net]\")\n",
    "    print(\"new filter load result =\", msg_new)\n",
    "\n",
    "    # 打印参数 shape 和 norm，确认确实加载了\n",
    "    print(\"\\n[ScenarioFilter.net parameter summary]\")\n",
    "    for n, p in scenario_filter.net.named_parameters():\n",
    "        print(f\"{n:20s} shape={tuple(p.shape)} norm={p.detach().norm().item():.6f}\")\n",
    "\n",
    "    return scenario_filter\n",
    "\n",
    "def build_new_dfl_for_test(args, data_path, target_nodes, models_s, device, quantiles):\n",
    "    \"\"\"\n",
    "    构建一个新框架下用于测试的 dfl 模型（single-node + SO）\n",
    "    \"\"\"\n",
    "    test_data = Combined_dataset_non_parametric(\n",
    "        data_path=data_path,\n",
    "        target_nodes=target_nodes,\n",
    "        flag=\"test\",\n",
    "        train_length=8760,\n",
    "        val_ratio=0.2,\n",
    "        seed=42,\n",
    "        y_real=True,\n",
    "    )\n",
    "\n",
    "    train_data = Combined_dataset_non_parametric(\n",
    "        data_path=data_path,\n",
    "        target_nodes=target_nodes,\n",
    "        flag=\"train\",\n",
    "        train_length=8760,\n",
    "        val_ratio=0.2,\n",
    "        seed=42,\n",
    "        y_real=True,\n",
    "    )\n",
    "    predictor = Multi_nonparametric_quantile_predictor(\n",
    "        node_models=copy.deepcopy(models_s),\n",
    "        scaler_y_map=train_data.scaler_y_map,\n",
    "        node_order=target_nodes,\n",
    "        quantiles=quantiles,   # <- 补上\n",
    "    ).to(device)\n",
    "\n",
    "    mgr_local = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "    optnet_DA = SingleNode_Reserve_SAA_DA_OptNet(\n",
    "        mgr=mgr_local,\n",
    "        N_scen=args.N_scen,\n",
    "        T=24,\n",
    "    ).to(device)\n",
    "    optnet_RT = SingleNode_Reserve_RT_OptNet(\n",
    "        mgr=mgr_local,\n",
    "        T=24,\n",
    "    ).to(device)\n",
    "\n",
    "    # set_seed(0)\n",
    "    # filter_module = ScenarioFilter(\n",
    "    #     args=args,\n",
    "    #     prob_type=\"single\",\n",
    "    #     T=24,\n",
    "    #     N_nodes=11,\n",
    "    #     K=int(args.N_scen),\n",
    "    #     K_rand=int(getattr(args, \"K_rand\", 0)),\n",
    "    #     hidden=128,\n",
    "    # ).to(device)\n",
    "\n",
    "    filter_module=result['dfl_trained'].scenario_filter\n",
    "\n",
    "    dfl = DFL_model_nonparametric_SingleNode(\n",
    "        mgr=mgr_local,\n",
    "        optnet_DA=optnet_DA,\n",
    "        optnet_RT=optnet_RT,\n",
    "        predictor=predictor,\n",
    "        quantiles=quantiles,   # <- 如果这个类构造函数也需要 quantiles，就要传\n",
    "        scenario_filter=filter_module,\n",
    "        n_scen=args.N_scen,\n",
    "        clamp_min=getattr(args, \"clamp_min\", 0.0),\n",
    "        solver=\"ECOS\",\n",
    "    ).to(device)\n",
    "\n",
    "    return dfl, test_data\n",
    "\n",
    "# ============================================================\n",
    "# 4) 直接运行：加载旧参数 -> 新框架测试\n",
    "# ============================================================\n",
    "def test_old_filter_ckpt_on_new_code(\n",
    "    args,\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    models_s,\n",
    "    device,\n",
    "    ckpt_path=\"checkpoints/nufilter_regular.pt\",\n",
    "    problem_mode=\"saa\",\n",
    "):\n",
    "    \"\"\"\n",
    "    直接测试：旧版 filter checkpoint 加载到新版 ScenarioFilter 后的效果\n",
    "    \"\"\"\n",
    "    print(\"\\n\" + \"=\" * 80)\n",
    "    print(\"Test old NuFilter checkpoint on NEW code\")\n",
    "    print(\"=\" * 80)\n",
    "\n",
    "    # 基本参数检查\n",
    "    K = int(args.N_scen)\n",
    "    K_rand = int(getattr(args, \"K_rand\", 0))\n",
    "    K_model = K - K_rand\n",
    "    if K_model <= 0:\n",
    "        raise ValueError(f\"K_model must be > 0, got N_scen={K}, K_rand={K_rand}\")\n",
    "\n",
    "    print(f\"K={K}, K_rand={K_rand}, K_model={K_model}\")\n",
    "    print(f\"eval_mode={getattr(args, 'eval_mode', 'soft')}\")\n",
    "    print(f\"eps_uniform={getattr(args, 'eps_uniform', None)}\")\n",
    "    print(f\"tau_gumbel={getattr(args, 'tau_gumbel', None)}\")\n",
    "\n",
    "    # 1) 构建新框架模型\n",
    "    dfl, test_data = build_new_dfl_for_test(\n",
    "        args=args,\n",
    "        data_path=data_path,\n",
    "        target_nodes=target_nodes,\n",
    "        models_s=models_s,\n",
    "        device=device,\n",
    "        quantiles=quantiles,\n",
    "    )\n",
    "\n",
    "    # 2) 加载旧 ckpt 到新 filter\n",
    "    load_old_ckpt_into_scenario_filter(\n",
    "        scenario_filter=dfl.scenario_filter,\n",
    "        ckpt_path=ckpt_path,\n",
    "        T=24,\n",
    "        hidden=128,\n",
    "        strict=True,\n",
    "        device=device,\n",
    "    )\n",
    "\n",
    "    # 3) 测试\n",
    "    losses = DFL_test(\n",
    "        dfl=dfl,\n",
    "        test_dataset=test_data,\n",
    "        args=args,\n",
    "        problem_mode=problem_mode,   # \"saa\" or \"dro\"\n",
    "        return_filter_aux=False,\n",
    "    )\n",
    "\n",
    "    mean_loss = losses.detach().float().mean().item()\n",
    "    std_loss = losses.detach().float().std().item()\n",
    "\n",
    "    print(\"\\n[RESULT] old ckpt loaded into new ScenarioFilter\")\n",
    "    print(f\"mean loss = {mean_loss:.6f}\")\n",
    "    print(f\"std  loss = {std_loss:.6f}\")\n",
    "    print(f\"n         = {len(losses)}\")\n",
    "\n",
    "    return {\n",
    "        \"dfl\": dfl,\n",
    "        \"test_data\": test_data,\n",
    "        \"losses\": losses,\n",
    "        \"mean_loss\": mean_loss,\n",
    "        \"std_loss\": std_loss,\n",
    "    }\n",
    "\n",
    "def test_fresh_new_filter_on_new_code(\n",
    "    args,\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    models_s,\n",
    "    device,\n",
    "    problem_mode=\"saa\",\n",
    "):\n",
    "    dfl, test_data = build_new_dfl_for_test(\n",
    "        args=args,\n",
    "        data_path=data_path,\n",
    "        target_nodes=target_nodes,\n",
    "        models_s=models_s,\n",
    "        device=device,\n",
    "        quantiles=quantiles\n",
    "    )\n",
    "\n",
    "    losses = DFL_test(\n",
    "        dfl=dfl,\n",
    "        test_dataset=test_data,\n",
    "        args=args,\n",
    "        problem_mode=problem_mode,\n",
    "        return_filter_aux=False,\n",
    "    )\n",
    "\n",
    "    mean_loss = losses.detach().float().mean().item()\n",
    "    std_loss = losses.detach().float().std().item()\n",
    "\n",
    "    print(\"\\n[RESULT] fresh new ScenarioFilter\")\n",
    "    print(f\"mean loss = {mean_loss:.6f}\")\n",
    "    print(f\"std  loss = {std_loss:.6f}\")\n",
    "    print(f\"n         = {len(losses)}\")\n",
    "\n",
    "    return {\n",
    "        \"dfl\": dfl,\n",
    "        \"test_data\": test_data,\n",
    "        \"losses\": losses,\n",
    "        \"mean_loss\": mean_loss,\n",
    "        \"std_loss\": std_loss,\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "124744b9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "================================================================================\n",
      "Test old NuFilter checkpoint on NEW code\n",
      "================================================================================\n",
      "K=20, K_rand=15, K_model=5\n",
      "eval_mode=discrete\n",
      "eps_uniform=0.1\n",
      "tau_gumbel=1.0\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_regular.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.491444\n",
      "0.bias               shape=(128,) norm=1.281435\n",
      "2.weight             shape=(5, 128) norm=1.340244\n",
      "2.bias               shape=(5,) norm=0.096176\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.22s/it, avg=3.85e+5, loss=2.36e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[RESULT] old ckpt loaded into new ScenarioFilter\n",
      "mean loss = 385111.687500\n",
      "std  loss = 154561.953125\n",
      "n         = 303\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# 建议保证这些参数和旧代码训练时一致\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.S_full = 200\n",
    "args.eval_mode = \"discrete\"   # 如果你想按旧代码 discrete eval\n",
    "args.eps_uniform = 0.10\n",
    "args.tau_gumbel = 1.0\n",
    "\n",
    "models_after = {node: result['dfl_trained'].predictor.models[str(node)] for node in target_nodes}\n",
    "result_old2new = test_old_filter_ckpt_on_new_code(\n",
    "    args=args,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_after,\n",
    "    device=device,\n",
    "    ckpt_path=\"checkpoints/nufilter_regular.pt\",\n",
    "    problem_mode=\"saa\",   # SO 对应这里用 \"saa\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "5ca2598a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "================================================================================\n",
      "Test old NuFilter checkpoint on NEW code\n",
      "================================================================================\n",
      "K=20, K_rand=15, K_model=5\n",
      "eval_mode=discrete\n",
      "eps_uniform=0.1\n",
      "tau_gumbel=1.0\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_init.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.460801\n",
      "0.bias               shape=(128,) norm=1.282785\n",
      "2.weight             shape=(5, 128) norm=1.303147\n",
      "2.bias               shape=(5,) norm=0.096323\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.47s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[RESULT] old ckpt loaded into new ScenarioFilter\n",
      "mean loss = 385645.843750\n",
      "std  loss = 154982.140625\n",
      "n         = 303\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# 建议保证这些参数和旧代码训练时一致\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.S_full = 200\n",
    "args.eval_mode = \"discrete\"   # 如果你想按旧代码 discrete eval\n",
    "args.eps_uniform = 0.10\n",
    "args.tau_gumbel = 1.0\n",
    "\n",
    "models_after = {node: result['dfl_trained'].predictor.models[str(node)] for node in target_nodes}\n",
    "result_old2new = test_old_filter_ckpt_on_new_code(\n",
    "    args=args,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_after,\n",
    "    device=device,\n",
    "    ckpt_path=\"checkpoints/nufilter_init.pt\",\n",
    "    problem_mode=\"saa\",   # SO 对应这里用 \"saa\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17a6ca71",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "================================================================================\n",
      "Test old NuFilter checkpoint on NEW code\n",
      "================================================================================\n",
      "K=20, K_rand=15, K_model=5\n",
      "eval_mode=discrete\n",
      "eps_uniform=0.1\n",
      "tau_gumbel=1.0\n",
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_non_parametric.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.500168\n",
      "0.bias               shape=(128,) norm=1.285351\n",
      "2.weight             shape=(5, 128) norm=1.344288\n",
      "2.bias               shape=(5,) norm=0.094388\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.58s/it, avg=3.82e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[RESULT] old ckpt loaded into new ScenarioFilter\n",
      "mean loss = 382372.625000\n",
      "std  loss = 149861.687500\n",
      "n         = 303\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "# 建议保证这些参数和旧代码训练时一致\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.S_full = 200\n",
    "args.eval_mode = \"discrete\"   # 如果你想按旧代码 discrete eval\n",
    "args.eps_uniform = 0.10\n",
    "args.tau_gumbel = 1.0\n",
    "\n",
    "models_after = {node: result['dfl_trained'].predictor.models[str(node)] for node in target_nodes}\n",
    "result_old2new = test_old_filter_ckpt_on_new_code(\n",
    "    args=args,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_after,\n",
    "    device=device,\n",
    "    ckpt_path=\"checkpoints/nufilter_non_parametric.pt\",\n",
    "    problem_mode=\"saa\",   # SO 对应这里用 \"saa\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "93b9c760",
   "metadata": {},
   "outputs": [],
   "source": [
    "import copy\n",
    "import torch\n",
    "\n",
    "# ============================================================\n",
    "# 1) 旧 NuFilter 定义（只用于加载 checkpoint 结构）\n",
    "# ============================================================\n",
    "class NuFilter(torch.nn.Module):\n",
    "    def __init__(self, T: int = 24, hidden: int = 128, K_model: int = 10):\n",
    "        super().__init__()\n",
    "        self.K_model = K_model\n",
    "        self.net = torch.nn.Sequential(\n",
    "            torch.nn.Linear(T, hidden),\n",
    "            torch.nn.ReLU(),\n",
    "            torch.nn.Linear(hidden, K_model),\n",
    "        )\n",
    "\n",
    "    def forward(self, Y_full_BST: torch.Tensor, tau: float = 1.0):\n",
    "        logits = self.net(Y_full_BST)   # (B,S,K_model)\n",
    "        logits = logits.transpose(1, 2) # (B,K_model,S)\n",
    "        w = torch.softmax(logits / max(float(tau), 1e-6), dim=-1)\n",
    "        return w\n",
    "\n",
    "# ============================================================\n",
    "# 2) 从旧 ckpt 加载并复制到新 ScenarioFilter.net\n",
    "# ============================================================\n",
    "def load_old_ckpt_into_scenario_filter(\n",
    "    scenario_filter,\n",
    "    ckpt_path=\"checkpoints/nufilter_regular.pt\",\n",
    "    T=24,\n",
    "    hidden=128,\n",
    "    strict=True,\n",
    "    device=None,\n",
    "):\n",
    "    \"\"\"\n",
    "    将旧版 NuFilter checkpoint 参数复制到新版 ScenarioFilter.net\n",
    "    \"\"\"\n",
    "    if device is None:\n",
    "        device = next(scenario_filter.parameters()).device\n",
    "\n",
    "    K_model = int(scenario_filter.K_model)\n",
    "    if K_model <= 0:\n",
    "        raise ValueError(f\"ScenarioFilter.K_model must be > 0, got {K_model}\")\n",
    "\n",
    "    old_model = NuFilter(T=T, hidden=hidden, K_model=K_model).to(device)\n",
    "\n",
    "    ckpt = torch.load(ckpt_path, map_location=device)\n",
    "    if isinstance(ckpt, dict) and \"state_dict\" in ckpt:\n",
    "        ckpt = ckpt[\"state_dict\"]\n",
    "\n",
    "    msg_old = old_model.load_state_dict(ckpt, strict=strict)\n",
    "    print(\"\\n[Load old NuFilter ckpt]\")\n",
    "    print(\"ckpt_path =\", ckpt_path)\n",
    "    print(\"old model load result =\", msg_old)\n",
    "\n",
    "    msg_new = scenario_filter.net.load_state_dict(old_model.net.state_dict(), strict=strict)\n",
    "    print(\"\\n[Copy old NuFilter.net -> new ScenarioFilter.net]\")\n",
    "    print(\"new filter load result =\", msg_new)\n",
    "\n",
    "    print(\"\\n[ScenarioFilter.net parameter summary]\")\n",
    "    for n, p in scenario_filter.net.named_parameters():\n",
    "        print(f\"{n:20s} shape={tuple(p.shape)} norm={p.detach().norm().item():.6f}\")\n",
    "\n",
    "    return scenario_filter\n",
    "\n",
    "# ============================================================\n",
    "# 3) 只测试已有 dfl_trained（不重建 predictor / dfl）\n",
    "# ============================================================\n",
    "def run_DFL_non_parametric_test(\n",
    "    args,\n",
    "    dfl_trained,\n",
    "    problem_mode,\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "):\n",
    "    import torch\n",
    "\n",
    "    eval_splits = tuple(s.lower() for s in (eval_splits or (\"test\",)))\n",
    "    if not all(s in (\"train\", \"test\") for s in eval_splits):\n",
    "        raise ValueError(f\"eval_splits must be subset of ('train','test'), got {eval_splits}\")\n",
    "\n",
    "    def make_dataset(flag):\n",
    "        return Combined_dataset_non_parametric(\n",
    "            data_path=data_path,\n",
    "            target_nodes=target_nodes,\n",
    "            flag=flag,\n",
    "            train_length=8760,\n",
    "            val_ratio=0.2,\n",
    "            seed=42,\n",
    "            y_real=True,\n",
    "        )\n",
    "\n",
    "    train_data = make_dataset(\"train\") if \"train\" in eval_splits else None\n",
    "    test_data = make_dataset(\"test\") if \"test\" in eval_splits else None\n",
    "\n",
    "    dfl_trained.eval()\n",
    "\n",
    "    out = {\n",
    "        \"eval_splits\": eval_splits,\n",
    "        \"problem_mode\": problem_mode,\n",
    "    }\n",
    "\n",
    "    if \"test\" in eval_splits:\n",
    "        set_seed(seed)\n",
    "        test_losses = DFL_test(dfl_trained, test_data, args, problem_mode=problem_mode)\n",
    "        out[\"test_losses_stage3_after\"] = test_losses\n",
    "        out[\"test_losses_after\"] = test_losses\n",
    "        out[\"test_losses_final\"] = test_losses\n",
    "\n",
    "        mean_test = float(torch.as_tensor(test_losses).detach().float().mean().item())\n",
    "        print(\"\\n[TEST] Final stage (stage3_after):\")\n",
    "        print(mean_test)\n",
    "\n",
    "    if \"train\" in eval_splits:\n",
    "        set_seed(seed)\n",
    "        train_losses = DFL_test(dfl_trained, train_data, args, problem_mode=problem_mode)\n",
    "        out[\"train_losses_stage3_after\"] = train_losses\n",
    "        out[\"train_losses_after\"] = train_losses\n",
    "        out[\"train_losses_final\"] = train_losses\n",
    "\n",
    "        mean_train = float(torch.as_tensor(train_losses).detach().float().mean().item())\n",
    "        print(\"\\n[TRAIN] Final stage (stage3_after):\")\n",
    "        print(mean_train)\n",
    "\n",
    "    return out\n",
    "\n",
    "# ============================================================\n",
    "# 4) 在现有 result['dfl_trained'] 上临时加载旧 ckpt，测试后恢复\n",
    "# ============================================================\n",
    "def test_old_filter_ckpt_on_existing_result(\n",
    "    args,\n",
    "    result,\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    device,\n",
    "    ckpt_path=\"checkpoints/nufilter_non_parametric.pt\",\n",
    "    problem_mode=\"saa\",\n",
    "    seed=0,\n",
    "):\n",
    "    \"\"\"\n",
    "    不重组 predictor，不 deepcopy 整个 dfl。\n",
    "    直接在 result['dfl_trained'] 上临时加载旧 ckpt，测试后恢复原 filter 参数。\n",
    "    \"\"\"\n",
    "    print(\"\\n\" + \"=\" * 80)\n",
    "    print(\"Test old NuFilter checkpoint on EXISTING dfl_trained\")\n",
    "    print(\"=\" * 80)\n",
    "\n",
    "    K = int(args.N_scen)\n",
    "    K_rand = int(getattr(args, \"K_rand\", 0))\n",
    "    K_model = K - K_rand\n",
    "    if K_model <= 0:\n",
    "        raise ValueError(f\"K_model must be > 0, got N_scen={K}, K_rand={K_rand}\")\n",
    "\n",
    "    print(f\"K={K}, K_rand={K_rand}, K_model={K_model}\")\n",
    "    print(f\"eval_mode={getattr(args, 'eval_mode', 'soft')}\")\n",
    "    print(f\"eps_uniform={getattr(args, 'eps_uniform', None)}\")\n",
    "    print(f\"tau_gumbel={getattr(args, 'tau_gumbel', None)}\")\n",
    "\n",
    "    dfl = result[\"dfl_trained\"]\n",
    "\n",
    "    # 只备份 filter 参数，避免 deepcopy 整个 dfl/cvxpylayer\n",
    "    orig_filter_state = copy.deepcopy(dfl.scenario_filter.state_dict())\n",
    "\n",
    "    try:\n",
    "        load_old_ckpt_into_scenario_filter(\n",
    "            scenario_filter=dfl.scenario_filter,\n",
    "            ckpt_path=ckpt_path,\n",
    "            T=24,\n",
    "            hidden=128,\n",
    "            strict=True,\n",
    "            device=device,\n",
    "        )\n",
    "\n",
    "        test_result = run_DFL_non_parametric_test(\n",
    "            args=args,\n",
    "            dfl_trained=dfl,\n",
    "            problem_mode=problem_mode,\n",
    "            data_path=data_path,\n",
    "            target_nodes=target_nodes,\n",
    "            seed=seed,\n",
    "            eval_splits=(\"test\",),\n",
    "        )\n",
    "\n",
    "        losses = test_result[\"test_losses_stage3_after\"]\n",
    "        mean_loss = losses.detach().float().mean().item()\n",
    "        std_loss = losses.detach().float().std().item()\n",
    "\n",
    "        print(\"\\n[RESULT] old ckpt loaded into existing dfl_trained\")\n",
    "        print(f\"mean loss = {mean_loss:.6f}\")\n",
    "        print(f\"std loss = {std_loss:.6f}\")\n",
    "        print(f\"n = {len(losses)}\")\n",
    "\n",
    "        return {\n",
    "            \"test_result\": test_result,\n",
    "            \"losses\": losses,\n",
    "            \"mean_loss\": mean_loss,\n",
    "            \"std_loss\": std_loss,\n",
    "        }\n",
    "\n",
    "    finally:\n",
    "        dfl.scenario_filter.load_state_dict(orig_filter_state, strict=True)\n",
    "        print(\"\\n[Restore] original ScenarioFilter parameters restored.\")\n",
    "\n",
    "# ============================================================\n",
    "# 5) 测试当前已有 dfl_trained（不加载旧 ckpt）\n",
    "# ============================================================\n",
    "def test_existing_result_filter(\n",
    "    args,\n",
    "    result,\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    problem_mode=\"saa\",\n",
    "    seed=0,\n",
    "):\n",
    "    print(\"\\n\" + \"=\" * 80)\n",
    "    print(\"Test EXISTING dfl_trained filter\")\n",
    "    print(\"=\" * 80)\n",
    "\n",
    "    test_result = run_DFL_non_parametric_test(\n",
    "        args=args,\n",
    "        dfl_trained=result[\"dfl_trained\"],\n",
    "        problem_mode=problem_mode,\n",
    "        data_path=data_path,\n",
    "        target_nodes=target_nodes,\n",
    "        seed=seed,\n",
    "        eval_splits=(\"test\",),\n",
    "    )\n",
    "\n",
    "    losses = test_result[\"test_losses_stage3_after\"]\n",
    "    mean_loss = losses.detach().float().mean().item()\n",
    "    std_loss = losses.detach().float().std().item()\n",
    "\n",
    "    print(\"\\n[RESULT] existing dfl_trained ScenarioFilter\")\n",
    "    print(f\"mean loss = {mean_loss:.6f}\")\n",
    "    print(f\"std loss = {std_loss:.6f}\")\n",
    "    print(f\"n = {len(losses)}\")\n",
    "\n",
    "    return {\n",
    "        \"test_result\": test_result,\n",
    "        \"losses\": losses,\n",
    "        \"mean_loss\": mean_loss,\n",
    "        \"std_loss\": std_loss,\n",
    "    }"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "667ee955",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "================================================================================\n",
      "Test EXISTING dfl_trained filter\n",
      "================================================================================\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.53s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Final stage (stage3_after):\n",
      "385826.0\n",
      "\n",
      "[RESULT] existing dfl_trained ScenarioFilter\n",
      "mean loss = 385826.000000\n",
      "std loss = 154831.718750\n",
      "n = 303\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "result_current = test_existing_result_filter(\n",
    "    args=args,\n",
    "    result=result,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    problem_mode=\"saa\",   # 或 \"so\"，按你 DFL_test 的兼容逻辑\n",
    "    seed=0,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "b86d96ab",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "================================================================================\n",
      "Test old NuFilter checkpoint on EXISTING dfl_trained\n",
      "================================================================================\n",
      "K=20, K_rand=15, K_model=5\n",
      "eval_mode=discrete\n",
      "eps_uniform=0.1\n",
      "tau_gumbel=0.1\n",
      "\n",
      "[Load old NuFilter ckpt]\n",
      "ckpt_path = checkpoints/nufilter_non_parametric.pt\n",
      "old model load result = <All keys matched successfully>\n",
      "\n",
      "[Copy old NuFilter.net -> new ScenarioFilter.net]\n",
      "new filter load result = <All keys matched successfully>\n",
      "\n",
      "[ScenarioFilter.net parameter summary]\n",
      "0.weight             shape=(128, 24) norm=6.500168\n",
      "0.bias               shape=(128,) norm=1.285351\n",
      "2.weight             shape=(5, 128) norm=1.344288\n",
      "2.bias               shape=(5,) norm=0.094388\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.25s/it, avg=3.85e+5, loss=2.38e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Final stage (stage3_after):\n",
      "385064.4375\n",
      "\n",
      "[RESULT] old ckpt loaded into existing dfl_trained\n",
      "mean loss = 385064.437500\n",
      "std loss = 153968.015625\n",
      "n = 303\n",
      "\n",
      "[Restore] original ScenarioFilter parameters restored.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "result_old2new = test_old_filter_ckpt_on_existing_result(\n",
    "    args=args,\n",
    "    result=result,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    ckpt_path=\"checkpoints/nufilter_non_parametric.pt\",\n",
    "    problem_mode=\"saa\",\n",
    "    seed=0,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "35145547",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387081.781250\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385416.656250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 385901.093750\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 385826.031250\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    }
   ],
   "source": [
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "045f8669",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "1.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.15s/it, avg=3.84e+5, loss=2.39e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[RESULT] fresh new ScenarioFilter\n",
      "mean loss = 383864.093750\n",
      "std  loss = 149998.984375\n",
      "n         = 303\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "result_fresh = test_fresh_new_filter_on_new_code(\n",
    "    args=args,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_after,\n",
    "    device=device,\n",
    "    problem_mode=\"saa\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2cfc094c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "1.0\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.06s/it, avg=3.83e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[RESULT] fresh new ScenarioFilter\n",
      "mean loss = 382919.812500\n",
      "std  loss = 149769.562500\n",
      "n         = 303\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "models_after = {node: result['dfl_trained'].predictor.models[str(node)] for node in target_nodes}\n",
    "result_fresh = test_fresh_new_filter_on_new_code(\n",
    "    args=args,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_after,\n",
    "    device=device,\n",
    "    problem_mode=\"saa\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b936c21",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "0.1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.21s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385932.21875\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.15s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386332.46875\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=0, lr=0.01, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 0it [00:00, ?it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 0.00 sec\n",
      "\n",
      " === total train time: 0.00 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.14s/it, avg=3.86e+5, loss=2.38e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386332.46875\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 0\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "args.eval_mode = \"discrete\"\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e5\n",
    "args.lambda_div_stage3=1e5\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=42,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "56002983",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[ScenarioFilter eval config]\n",
      "eval_mode = discrete\n",
      "avoid_rand_duplicate = False\n",
      "K_rand 15\n",
      "K_model 5\n",
      "0.1\n",
      "1\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.21s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385932.21875\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.15s/it, avg=3.87e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "387339.03125\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=0, lr=0.01, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 0it [00:00, ?it/s]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 0.00 sec\n",
      "\n",
      " === total train time: 0.00 sec ===\n",
      "[DFL_test] eval_mode=discrete, avoid_rand_duplicate=False\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.15s/it, avg=3.87e+5, loss=2.38e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "387339.03125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 15\n",
    "args.filter_epochs = 0\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "args.eval_mode = \"discrete\"\n",
    "args.tau_gumbel = 1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e5\n",
    "args.lambda_div_stage3=1e5\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=42,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f1991c4b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:04<00:00,  2.04it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.39s/it, avg=3.87e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387081.78125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:15<00:00, 65.08s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 195.41 sec\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.12s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385416.65625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=5, lr=0.005, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.11s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "386208.5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 5/5 [05:54<00:00, 70.90s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 354.52 sec\n",
      "\n",
      " === total train time: 592.25 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.59s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385949.71875\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387081.781250\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385416.656250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 386208.531250\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 385949.718750\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 5e-3\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 5\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div_stage3=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 1\n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ce6a635e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.48s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385416.65625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.42s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386101.59375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:15<00:00, 75.48s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 75.48 sec\n",
      "\n",
      " === total train time: 75.48 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.36s/it, avg=3.86e+5, loss=2.38e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385978.28125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 1\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e4 #entropy\n",
    "args.lambda_div_stage3=1e4\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "400ef3f8",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-123.3125"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "385978.28125-386101.59375"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "9b5aecf9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.31s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385416.65625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.32s/it, avg=3.89e+5, loss=2.39e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "388607.625\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:12<00:00, 72.76s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 72.77 sec\n",
      "\n",
      " === total train time: 72.77 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.47s/it, avg=3.88e+5, loss=2.39e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "388460.59375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 1\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 1\n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e4 #entropy\n",
    "args.lambda_div_stage3=1e4\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "e3809c60",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-147.03125"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "388460.59375-388607.625"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a6985e98",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.55s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385416.65625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.36s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386101.59375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:23<00:00, 83.31s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 83.32 sec\n",
      "\n",
      " === total train time: 83.32 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.58s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386113.53125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.36s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385416.65625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.46s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386101.59375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:20<00:00, 80.94s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 80.95 sec\n",
      "\n",
      " === total train time: 80.95 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.26s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386124.40625\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.30s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385416.65625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.36s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386101.59375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:19<00:00, 79.49s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 79.50 sec\n",
      "\n",
      " === total train time: 79.50 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.52s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "385978.28125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.43s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385416.65625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:26<00:00,  2.69s/it, avg=3.86e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386101.59375\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=1, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 1/1 [01:16<00:00, 76.20s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 76.21 sec\n",
      "\n",
      " === total train time: 76.21 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:25<00:00,  2.57s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386049.34375\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 1\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e4 #entropy\n",
    "args.lambda_div_stage3=1e6\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")\n",
    "\n",
    "args.lambda_div_stage3=1e5\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")\n",
    "\n",
    "\n",
    "args.lambda_div_stage3=1e4\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")\n",
    "\n",
    "\n",
    "args.lambda_div_stage3=1e3\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92593bbc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:22<00:00,  2.28s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before filter training):\n",
      "385613.1875\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:24<00:00,  2.40s/it, avg=3.87e+5, loss=2.38e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter BEFORE training:\n",
      "386689.28125\n",
      "\n",
      " ---> Stage B: train Filter only (epochs=3, lr=0.005, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [03:39<00:00, 73.22s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 219.66 sec\n",
      "\n",
      " === total train time: 219.66 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:23<00:00,  2.30s/it, avg=3.87e+5, loss=2.38e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386646.3125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.filter_epochs = 3\n",
    "args.filter_lr = 5e-3\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "args.lambda_div=1e4 #entropy\n",
    "args.lambda_div_stage3=1e6\n",
    "result_stage3_only = run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev=result,\n",
    "    args=args,\n",
    "    problem_mode=\"so\",          # 跟你原来一致\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    device=device,\n",
    "    seed=1,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True, \n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4fbb460e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-81.8125"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "386607.46875-386689.28125"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "051b95eb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:05<00:00,  1.87it/s, avg=4.26e+5, loss=2.6e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Deterministic baseline (before DFL training):\n",
      "425659.3125\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.08s/it, avg=3.87e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (before DFL training):\n",
      "387081.78125\n",
      "\n",
      " ---> Stage A: train DFL with RANDOM selector (epochs=3, dfl_lr=1e-06, bs=32)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (dfl): 100%|██████████| 3/3 [03:21<00:00, 67.08s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage A done] time: 201.25 sec\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.13s/it, avg=3.85e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter cost AFTER DFL training:\n",
      "385416.65625\n",
      "\n",
      " ---> Stage B: switch to learnable filter, train Filter only (epochs=3, lr=0.01, bs=32)\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.16s/it, avg=3.86e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training:\n",
      "386208.5\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Train (filter_only): 100%|██████████| 3/3 [03:26<00:00, 68.95s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " ---> [Stage B done] time: 206.86 sec\n",
      "\n",
      " === total train time: 450.94 sec ===\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.20s/it, avg=3.86e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] ScenarioFilter AFTER training:\n",
      "386012.75\n",
      "\n",
      "============================================================\n",
      "  Final Results Summary (SEPERATE | SINGLE | SO)\n",
      "============================================================\n",
      " -> Deterministic Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 425659.312500\n",
      " -> Random Baseline (before DFL training):\n",
      "    [TEST ] Mean Loss: 387081.781250\n",
      " -> Random Filter AFTER DFL training:\n",
      "    [TEST ] Mean Loss: 385416.656250\n",
      " -> Fresh ScenarioFilter BEFORE training:\n",
      "    [TEST ] Mean Loss: 386208.531250\n",
      " -> ScenarioFilter AFTER training:\n",
      "    [TEST ] Mean Loss: 386012.781250\n",
      "============================================================\n",
      "\n",
      "Full model pickled: ../Result/Non_parametric/KL/DFL_model_trained_seperate_single_SO.pkl\n",
      "Saved at: ../Result/Non_parametric/KL/dfl_result_seperate_single_SO.pkl\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode='single'\n",
    "problem_mode='so'\n",
    "forecasting_mode='seperate'\n",
    "\n",
    "\n",
    "mgr=IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "if optimization_mode=='multi':\n",
    "    args.Lmin=mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax=mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value=1000\n",
    "else:\n",
    "    args.Lmin=Lmin_total\n",
    "    args.Lmax=Lmax_total\n",
    "    args.eps_value=1000\n",
    "\n",
    "\n",
    "mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "\n",
    "# 写入大一统需要的 args 配置边界\n",
    "if optimization_mode == 'multi':\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)  # dro 会用到\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)  # dro 会用到\n",
    "else:\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "\n",
    "args.run_stage2 = True\n",
    "args.run_stage3 = True\n",
    "\n",
    "args.dfl_epochs = 3    # Stage 3 (联合微调) 轮数 (端到端微调极耗时，一般1-3轮即收敛)\n",
    "args.dfl_lr = 1e-6\n",
    "args.filter_lr = 1e-2\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.filter_epochs = 3\n",
    "args.clip_predictor = 1.0\n",
    "args.S_full= 200\n",
    "args.div_type='kl'\n",
    "#args.lambda_div=1e3 inner\n",
    "#args.lambda_div=1e4 #kl\n",
    "#entropy 1e3\n",
    "args.lambda_div=1e3 #entropy\n",
    "\n",
    "args.tau_gumbel = 0.1 \n",
    "args.eps_uniform = 0.1\n",
    "result=run_DFL_non_parametric_separate(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes, \n",
    "    models_s=models_s_non_parametric,\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    ")\n",
    "\n",
    "save_run_result(result,forecasting_mode,out_dir=\"../Result/Non_parametric/KL\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6413a834",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1000000.0"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lambda_div"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "248a8449",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000.0"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "args.lambda_div"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "623fc5fb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "10000.0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "getattr(args, \"lambda_div\", 1e5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "e616db9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "lambda_div = float(getattr(args, \"lambda_div_stage3\", getattr(args, \"lambda_div\", 1e5)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4e30a3bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_only_stage3_from_result_non_parametric_separate(\n",
    "    result_prev,\n",
    "    args,\n",
    "    problem_mode,  # \"so\" / \"dro\"\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    device,\n",
    "    seed=0,\n",
    "    eval_splits=(\"test\",),\n",
    "    reinit_filter=True,\n",
    "    eval_flags=(True, True, True),  # (random_before, stage3_before, stage3_after)\n",
    "):\n",
    "    import copy\n",
    "    import time\n",
    "    import torch\n",
    "\n",
    "    def is_so(x):\n",
    "        return str(x).lower() in {\"so\", \"saa\"}\n",
    "\n",
    "    eval_splits = tuple(s.lower() for s in (eval_splits or (\"test\",)))\n",
    "    if not all(s in (\"train\", \"test\") for s in eval_splits):\n",
    "        raise ValueError(f\"eval_splits must be subset of ('train','test'), got {eval_splits}\")\n",
    "\n",
    "    if len(eval_flags) != 3:\n",
    "        raise ValueError(\n",
    "            \"eval_flags must be a 3-tuple: \"\n",
    "            \"(random_before, stage3_before, stage3_after)\"\n",
    "        )\n",
    "    eval_random_before, eval_stage3_before, eval_stage3_after = map(bool, eval_flags)\n",
    "\n",
    "    # -------------------------\n",
    "    # local args\n",
    "    # -------------------------\n",
    "    local_args = copy.deepcopy(args)\n",
    "\n",
    "    # -------------------------\n",
    "    # dataset\n",
    "    # -------------------------\n",
    "    def make_dataset(flag):\n",
    "        return Combined_dataset_non_parametric(\n",
    "            data_path=data_path,\n",
    "            target_nodes=target_nodes,\n",
    "            flag=flag,\n",
    "            train_length=8760,\n",
    "            val_ratio=0.2,\n",
    "            seed=42,\n",
    "            y_real=True,\n",
    "        )\n",
    "\n",
    "    train_data = make_dataset(\"train\")\n",
    "    test_data = make_dataset(\"test\")\n",
    "\n",
    "    # -------------------------\n",
    "    # load previous trained model\n",
    "    # -------------------------\n",
    "    if \"dfl_trained\" not in result_prev:\n",
    "        raise KeyError(\"result_prev must contain key 'dfl_trained'\")\n",
    "\n",
    "    dfl = result_prev[\"dfl_trained\"].to(device)\n",
    "\n",
    "    optimization_mode = str(result_prev.get(\"optimization_mode\", \"multi\")).lower()\n",
    "    multi = optimization_mode in {\"multi\", \"multinode\"}\n",
    "\n",
    "    mode_str = \"SO\" if is_so(problem_mode) else \"DRO\"\n",
    "\n",
    "    # -------------------------\n",
    "    # scenario filter config\n",
    "    # -------------------------\n",
    "    local_args.S_full = int(getattr(local_args, \"S_full\", result_prev.get(\"S_full\", 50)))\n",
    "    K = int(getattr(local_args, \"N_scen\", result_prev.get(\"N_scen\", 50)))\n",
    "    K_rand = int(getattr(local_args, \"K_rand\", result_prev.get(\"K_rand\", 0)))\n",
    "\n",
    "    if K_rand > K:\n",
    "        raise ValueError(f\"K_rand({K_rand}) must be <= N_scen({K}).\")\n",
    "\n",
    "    # optionally reinit filter before stage3\n",
    "    if reinit_filter:\n",
    "        set_seed(seed)\n",
    "        dfl.scenario_filter = ScenarioFilter(\n",
    "            args=local_args,\n",
    "            prob_type=\"multi\" if multi else \"single\",\n",
    "            T=24,\n",
    "            N_nodes=11,\n",
    "            K=K,\n",
    "            K_rand=K_rand,\n",
    "            hidden=128,\n",
    "        ).to(device)\n",
    "\n",
    "    # -------------------------\n",
    "    # helper: evaluate on requested splits\n",
    "    # -------------------------\n",
    "    def eval_on_splits(model, stage_tag):\n",
    "        out = {}\n",
    "        if \"test\" in eval_splits:\n",
    "            set_seed(seed)\n",
    "            out[f\"test_losses_{stage_tag}\"] = DFL_test(\n",
    "                model, test_data, local_args, problem_mode=problem_mode\n",
    "            )\n",
    "        if \"train\" in eval_splits:\n",
    "            set_seed(seed)\n",
    "            out[f\"train_losses_{stage_tag}\"] = DFL_test(\n",
    "                model, train_data, local_args, problem_mode=problem_mode\n",
    "            )\n",
    "        return out\n",
    "\n",
    "    def print_mean_loss(title, eval_dict, key):\n",
    "        if key in eval_dict and eval_dict[key] is not None:\n",
    "            val = eval_dict[key]\n",
    "            if hasattr(val, \"detach\"):\n",
    "                mean_val = val.detach().float().mean().item()\n",
    "            else:\n",
    "                mean_val = float(torch.as_tensor(val).float().mean().item())\n",
    "            print(f\"\\n[TEST] {title}:\")\n",
    "            print(mean_val)\n",
    "\n",
    "    # -------------------------\n",
    "    # train setup\n",
    "    # -------------------------\n",
    "    bs_train = int(getattr(local_args, \"dfl_batch_size\", getattr(local_args, \"batch_size\", 8)))\n",
    "    old_bs = getattr(local_args, \"batch_size\", None)\n",
    "    local_args.batch_size = bs_train\n",
    "\n",
    "    local_args.epochs = int(getattr(local_args, \"filter_epochs\", 10))\n",
    "    local_args.lr = float(getattr(local_args, \"filter_lr\", 1e-3))\n",
    "    lambda_div = float(getattr(local_args, \"lambda_div_stage3\", getattr(local_args, \"lambda_div\", 1e5)))\n",
    "\n",
    "    stage1_eval = {}\n",
    "    stage3_before_eval = {}\n",
    "    stage3_eval = {}\n",
    "\n",
    "    # -------------------------\n",
    "    # stage1_after: random baseline\n",
    "    # -------------------------\n",
    "    if eval_random_before:\n",
    "        current_filter = dfl.scenario_filter\n",
    "        random_filter = RandomScenarioSelector(n_scen=int(K)).to(device)\n",
    "\n",
    "        dfl.scenario_filter = random_filter\n",
    "        stage1_eval = eval_on_splits(dfl, \"stage1_after\")\n",
    "\n",
    "        if \"test\" in eval_splits:\n",
    "            print_mean_loss(\n",
    "                \"Random filter baseline (before filter training)\",\n",
    "                stage1_eval,\n",
    "                \"test_losses_stage1_after\",\n",
    "            )\n",
    "\n",
    "        # restore current/fresh filter\n",
    "        dfl.scenario_filter = current_filter\n",
    "\n",
    "    # -------------------------\n",
    "    # stage3_before\n",
    "    # -------------------------\n",
    "    if eval_stage3_before:\n",
    "        stage3_before_eval = eval_on_splits(dfl, \"stage3_before\")\n",
    "\n",
    "        if \"test\" in eval_splits:\n",
    "            print_mean_loss(\n",
    "                \"ScenarioFilter BEFORE training\",\n",
    "                stage3_before_eval,\n",
    "                \"test_losses_stage3_before\",\n",
    "            )\n",
    "\n",
    "    # -------------------------\n",
    "    # freeze predictor, train filter only\n",
    "    # -------------------------\n",
    "    for p in dfl.predictor.parameters():\n",
    "        p.requires_grad_(False)\n",
    "\n",
    "    if getattr(dfl, \"scenario_filter\", None) is not None:\n",
    "        for p in dfl.scenario_filter.parameters():\n",
    "            p.requires_grad_(True)\n",
    "\n",
    "    # -------------------------\n",
    "    # stage3 training\n",
    "    # -------------------------\n",
    "    set_seed(seed)\n",
    "    t0_total = time.time()\n",
    "    t0_s3 = time.time()\n",
    "\n",
    "    print(\n",
    "        f\"\\n ---> Stage B: train Filter only \"\n",
    "        f\"(epochs={local_args.epochs}, lr={local_args.lr}, bs={local_args.batch_size})\"\n",
    "    )\n",
    "\n",
    "    dfl, train_logs_stage3 = DFL_train(\n",
    "        dfl,\n",
    "        train_data,\n",
    "        local_args,\n",
    "        problem_mode=problem_mode,\n",
    "        train_mode=\"filter_only\",\n",
    "        lambda_div=lambda_div,\n",
    "    )\n",
    "\n",
    "    time_stage3 = time.time() - t0_s3\n",
    "    train_time_sec_total = time.time() - t0_total\n",
    "\n",
    "    print(f\" ---> [Stage B done] time: {time_stage3:.2f} sec\")\n",
    "    print(f\"\\n === total train time: {train_time_sec_total:.2f} sec ===\")\n",
    "\n",
    "    # -------------------------\n",
    "    # stage3_after\n",
    "    # -------------------------\n",
    "    if eval_stage3_after:\n",
    "        stage3_eval = eval_on_splits(dfl, \"stage3_after\")\n",
    "\n",
    "        if \"test\" in eval_splits:\n",
    "            print_mean_loss(\n",
    "                \"ScenarioFilter AFTER training\",\n",
    "                stage3_eval,\n",
    "                \"test_losses_stage3_after\",\n",
    "            )\n",
    "\n",
    "    # restore args\n",
    "    if old_bs is not None:\n",
    "        local_args.batch_size = old_bs\n",
    "\n",
    "    # -------------------------\n",
    "    # result\n",
    "    # -------------------------\n",
    "    result = {\n",
    "        \"optimization_mode\": \"multi\" if multi else \"single\",\n",
    "        \"problem_mode\": mode_str,\n",
    "        \"eval_splits\": eval_splits,\n",
    "        \"eval_flags\": tuple(eval_flags),\n",
    "        \"dfl_trained\": dfl,\n",
    "        \"train_logs_stage3\": train_logs_stage3,\n",
    "        \"time_stage3_sec\": float(time_stage3),\n",
    "        \"train_time_sec_total\": float(train_time_sec_total),\n",
    "        \"train_batch_size_used\": int(bs_train),\n",
    "        \"N_scen\": int(K),\n",
    "        \"S_full\": int(local_args.S_full),\n",
    "        \"K_rand\": int(K_rand),\n",
    "        \"reinit_filter\": bool(reinit_filter),\n",
    "    }\n",
    "\n",
    "    result.update(stage1_eval)\n",
    "    result.update(stage3_before_eval)\n",
    "    result.update(stage3_eval)\n",
    "\n",
    "    # -------------------------\n",
    "    # explicit aliases\n",
    "    # -------------------------\n",
    "    if \"test_losses_stage1_after\" in result:\n",
    "        result[\"test_losses_random_baseline\"] = result[\"test_losses_stage1_after\"]\n",
    "    if \"train_losses_stage1_after\" in result:\n",
    "        result[\"train_losses_random_baseline\"] = result[\"train_losses_stage1_after\"]\n",
    "\n",
    "    if \"test_losses_stage3_before\" in result:\n",
    "        result[\"test_losses_scenario_filter_before_training\"] = result[\"test_losses_stage3_before\"]\n",
    "    if \"train_losses_stage3_before\" in result:\n",
    "        result[\"train_losses_scenario_filter_before_training\"] = result[\"train_losses_stage3_before\"]\n",
    "\n",
    "    if \"test_losses_stage3_after\" in result:\n",
    "        result[\"test_losses_scenario_filter_after_training\"] = result[\"test_losses_stage3_after\"]\n",
    "    if \"train_losses_stage3_after\" in result:\n",
    "        result[\"train_losses_scenario_filter_after_training\"] = result[\"train_losses_stage3_after\"]\n",
    "\n",
    "    # -------------------------\n",
    "    # backward-compatible aliases\n",
    "    # -------------------------\n",
    "    if \"test_losses_stage3_before\" in result:\n",
    "        result[\"test_losses_before_filter_training\"] = result[\"test_losses_stage3_before\"]\n",
    "    if \"train_losses_stage3_before\" in result:\n",
    "        result[\"train_losses_before_filter_training\"] = result[\"train_losses_stage3_before\"]\n",
    "\n",
    "    if \"test_losses_stage3_after\" in result:\n",
    "        result[\"test_losses_after_filter_training\"] = result[\"test_losses_stage3_after\"]\n",
    "    if \"train_losses_stage3_after\" in result:\n",
    "        result[\"train_losses_after_filter_training\"] = result[\"train_losses_stage3_after\"]\n",
    "\n",
    "    # -------------------------\n",
    "    # generic before / after aliases\n",
    "    # -------------------------\n",
    "    if \"train\" in eval_splits:\n",
    "        if \"train_losses_stage1_after\" in result:\n",
    "            result[\"train_losses_before\"] = result[\"train_losses_stage1_after\"]\n",
    "        if \"train_losses_stage3_after\" in result:\n",
    "            result[\"train_losses_after\"] = result[\"train_losses_stage3_after\"]\n",
    "\n",
    "    if \"test\" in eval_splits:\n",
    "        if \"test_losses_stage1_after\" in result:\n",
    "            result[\"test_losses_before\"] = result[\"test_losses_stage1_after\"]\n",
    "        if \"test_losses_stage3_after\" in result:\n",
    "            result[\"test_losses_after\"] = result[\"test_losses_stage3_after\"]\n",
    "\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "79b318e4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:21<00:00,  2.13s/it, avg=3.87e+5, loss=2.37e+5]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Random filter baseline (test):\n",
      "387081.78125\n",
      "K_rand 10\n",
      "K_model 10\n",
      "0.1\n",
      "0.1\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Test: 100%|██████████| 10/10 [00:20<00:00,  2.07s/it, avg=3.88e+5, loss=2.37e+5]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "[TEST] Fresh ScenarioFilter BEFORE training (test):\n",
      "387889.15625\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\n"
     ]
    }
   ],
   "source": [
    "optimization_mode = 'single'\n",
    "problem_mode = 'so'\n",
    "forecasting_mode = 'joint_untrained_compare'\n",
    "\n",
    "# manager / args boundary setup\n",
    "if optimization_mode == 'multi':\n",
    "    mgr = IEEE14_Reserve_SO_Manager_MultiNode(args)\n",
    "    args.Lmin = mgr.map_11load_to_14bus(Lmin)\n",
    "    args.Lmax = mgr.map_11load_to_14bus(Lmax)\n",
    "    args.eps_value = 1000\n",
    "else:\n",
    "    mgr = IEEE14_Reserve_SO_Manager_SingleNode(args)\n",
    "    args.Lmin = Lmin_total\n",
    "    args.Lmax = Lmax_total\n",
    "    args.eps_value = 1000\n",
    "\n",
    "# untrained compare 需要的关键参数\n",
    "args.N_scen = 20\n",
    "args.K_rand = 10\n",
    "args.S_full = 200\n",
    "args.div_type = 'kl'\n",
    "args.lambda_div = 1e3   # 虽然这个函数里通常不会训练，但保留也没问题\n",
    "\n",
    "\n",
    "args.tau_gumbel = 0.1   # Gumbel Softmax 温度\n",
    "args.eps_uniform = 0.1 # 防震荡平滑参数\n",
    "\n",
    "result = run_DFL_non_parametric_joint_untrained_compare(\n",
    "    args=args,\n",
    "    optimization_mode=optimization_mode,\n",
    "    problem_mode=problem_mode,\n",
    "    quantiles=quantiles,\n",
    "    data_path=data_path,\n",
    "    target_nodes=target_nodes,\n",
    "    models_s=models_s_non_parametric,           # joint non-parametric model\n",
    "    device=device,\n",
    "    seed=0,\n",
    "    eval_split=\"test\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "0bebc06e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def run_DFL_non_parametric_joint_untrained_compare(\n",
    "    args,\n",
    "    problem_mode,          # \"so\" / \"dro\"\n",
    "    optimization_mode,     # \"single\" / \"multi\"\n",
    "    quantiles,\n",
    "    data_path,\n",
    "    target_nodes,\n",
    "    models_s,\n",
    "    device,\n",
    "    seed=0,\n",
    "    eval_split=\"test\",     # \"test\" / \"train\"\n",
    "):\n",
    "    import copy\n",
    "    import torch\n",
    "\n",
    "    def is_so(x):\n",
    "        return str(x).lower() in {\"so\", \"saa\"}\n",
    "\n",
    "    def is_multi(x):\n",
    "        return str(x).lower() in {\"multi\", \"multinode\"}\n",
    "\n",
    "    def mean_tensor(x):\n",
    "        if hasattr(x, \"detach\"):\n",
    "            return x.detach().float().mean().item()\n",
    "        return float(torch.as_tensor(x).float().mean().item())\n",
    "\n",
    "    eval_split = str(eval_split).lower()\n",
    "    if eval_split not in (\"train\", \"test\"):\n",
    "        raise ValueError(f\"eval_split must be 'train' or 'test', got {eval_split}\")\n",
    "\n",
    "    # -------- dataset --------\n",
    "    def make_dataset(flag):\n",
    "        return Combined_dataset_non_parametric(\n",
    "            data_path=data_path,\n",
    "            target_nodes=target_nodes,\n",
    "            flag=flag,\n",
    "            train_length=8760,\n",
    "            val_ratio=0.2,\n",
    "            seed=42,\n",
    "            y_real=True,\n",
    "        )\n",
    "\n",
    "    train_data = make_dataset(\"train\")\n",
    "    test_data = make_dataset(\"test\")\n",
    "    eval_data = train_data if eval_split == \"train\" else test_data\n",
    "\n",
    "    # -------- predictor --------\n",
    "    # 基于你现有 separate 代码，沿用 Multi_nonparametric_quantile_predictor\n",
    "    def build_predictor(node_models, scaler_y_map, target_nodes_, quantiles_):\n",
    "        return Multi_nonparametric_quantile_predictor(\n",
    "            node_models=copy.deepcopy(node_models),\n",
    "            scaler_y_map=scaler_y_map,\n",
    "            node_order=target_nodes_,\n",
    "            quantiles=quantiles_,\n",
    "        ).to(device)\n",
    "\n",
    "    predictor_random = build_predictor(models_s, train_data.scaler_y_map, target_nodes, quantiles)\n",
    "    predictor_scen = build_predictor(models_s, train_data.scaler_y_map, target_nodes, quantiles)\n",
    "\n",
    "    # -------- pick manager/optnet/DFL --------\n",
    "    multi = is_multi(optimization_mode)\n",
    "\n",
    "    if multi:\n",
    "        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_MultiNode, IEEE14_Reserve_DRO_Manager_MultiNode\n",
    "        SAA_DA_OptNet, DRO_DA_OptNet = MultiNode_Reserve_SAA_DA_OptNet, MultiNode_Reserve_DRO_DA_OptNet\n",
    "        RT_OptNet = MultiNode_Reserve_RT_OptNet\n",
    "        DFL_SO_Class, DFL_DRO_Class = DFL_model_nonparametric_MultiNode, DFL_model_nonparametric_DRO_MultiNode\n",
    "    else:\n",
    "        SO_Manager, DRO_Manager = IEEE14_Reserve_SO_Manager_SingleNode, IEEE14_Reserve_DRO_Manager_SingleNode\n",
    "        SAA_DA_OptNet, DRO_DA_OptNet = SingleNode_Reserve_SAA_DA_OptNet, SingleNode_Reserve_DRO_DA_OptNet\n",
    "        RT_OptNet = SingleNode_Reserve_RT_OptNet\n",
    "        DFL_SO_Class, DFL_DRO_Class = DFL_model_nonparametric_SingleNode, DFL_model_nonparametric_DRO_SingleNode\n",
    "\n",
    "    if is_so(problem_mode):\n",
    "        mgr_local = SO_Manager(args)\n",
    "        optnet_DA = SAA_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)\n",
    "        DFLClass = DFL_SO_Class\n",
    "        mode_str = \"SO\"\n",
    "    else:\n",
    "        mgr_local = DRO_Manager(args)\n",
    "        optnet_DA = DRO_DA_OptNet(mgr=mgr_local, N_scen=args.N_scen, T=24).to(device)\n",
    "        DFLClass = DFL_DRO_Class\n",
    "        mode_str = \"DRO\"\n",
    "\n",
    "    optnet_RT = RT_OptNet(mgr=mgr_local, T=24).to(device)\n",
    "\n",
    "    # -------- filter config --------\n",
    "    args.S_full = int(getattr(args, \"S_full\", 50))\n",
    "    K = int(getattr(args, \"N_scen\", 50))\n",
    "    K_rand = int(getattr(args, \"K_rand\", 0))\n",
    "    if K_rand > K:\n",
    "        raise ValueError(f\"K_rand({K_rand}) must be <= N_scen({K}).\")\n",
    "\n",
    "    clamp_min = float(getattr(args, \"clamp_min\", 0.0))\n",
    "    solver = getattr(args, \"solver\", \"ECOS\")\n",
    "\n",
    "    # -------- 1) random filter baseline --------\n",
    "    dfl_random = DFLClass(\n",
    "        mgr=mgr_local,\n",
    "        optnet_DA=optnet_DA,\n",
    "        optnet_RT=optnet_RT,\n",
    "        predictor=predictor_random,\n",
    "        quantiles=quantiles,\n",
    "        scenario_filter=RandomScenarioSelector(n_scen=int(args.N_scen)).to(device),\n",
    "        n_scen=args.N_scen,\n",
    "        clamp_min=clamp_min,\n",
    "        solver=solver,\n",
    "    ).to(device)\n",
    "\n",
    "    set_seed(seed)\n",
    "    losses_random = DFL_test(\n",
    "        dfl_random,\n",
    "        eval_data,\n",
    "        args,\n",
    "        problem_mode=problem_mode,\n",
    "    )\n",
    "\n",
    "    print(f\"\\n[TEST] Random filter baseline ({eval_split}):\")\n",
    "    print(mean_tensor(losses_random))\n",
    "\n",
    "    # -------- 2) fresh ScenarioFilter BEFORE training --------\n",
    "    fresh_filter = ScenarioFilter(\n",
    "        args=args,\n",
    "        prob_type=\"multi\" if multi else \"single\",\n",
    "        T=24,\n",
    "        N_nodes=11,\n",
    "        K=K,\n",
    "        K_rand=K_rand,\n",
    "        hidden=128,\n",
    "    ).to(device)\n",
    "\n",
    "    dfl_scen_untrained = DFLClass(\n",
    "        mgr=mgr_local,\n",
    "        optnet_DA=optnet_DA,\n",
    "        optnet_RT=optnet_RT,\n",
    "        predictor=predictor_scen,\n",
    "        quantiles=quantiles,\n",
    "        scenario_filter=fresh_filter,\n",
    "        n_scen=args.N_scen,\n",
    "        clamp_min=clamp_min,\n",
    "        solver=solver,\n",
    "    ).to(device)\n",
    "\n",
    "    set_seed(seed)\n",
    "    losses_scen_untrained = DFL_test(\n",
    "        dfl_scen_untrained,\n",
    "        eval_data,\n",
    "        args,\n",
    "        problem_mode=problem_mode,\n",
    "    )\n",
    "\n",
    "    print(f\"\\n[TEST] Fresh ScenarioFilter BEFORE training ({eval_split}):\")\n",
    "    print(mean_tensor(losses_scen_untrained))\n",
    "\n",
    "    # -------- result --------\n",
    "    result = {\n",
    "        \"optimization_mode\": \"multi\" if multi else \"single\",\n",
    "        \"problem_mode\": mode_str,\n",
    "        \"eval_split\": eval_split,\n",
    "        \"N_scen\": int(args.N_scen),\n",
    "        \"S_full\": int(args.S_full),\n",
    "        \"K_rand\": int(K_rand),\n",
    "        \"dfl_random\": dfl_random,\n",
    "        \"dfl_scen_untrained\": dfl_scen_untrained,\n",
    "        \"losses_random_baseline\": losses_random,\n",
    "        \"losses_scenario_filter_before_training\": losses_scen_untrained,\n",
    "        \"random_baseline_mean\": mean_tensor(losses_random),\n",
    "        \"scenario_filter_before_training_mean\": mean_tensor(losses_scen_untrained),\n",
    "    }\n",
    "\n",
    "    if eval_split == \"test\":\n",
    "        result[\"test_losses_random_baseline\"] = losses_random\n",
    "        result[\"test_losses_scenario_filter_before_training\"] = losses_scen_untrained\n",
    "        result[\"test_random_baseline_mean\"] = mean_tensor(losses_random)\n",
    "        result[\"test_scenario_filter_before_training_mean\"] = mean_tensor(losses_scen_untrained)\n",
    "\n",
    "        # backward-compatible aliases\n",
    "        result[\"test_losses_before\"] = losses_random\n",
    "        result[\"test_losses_after\"] = losses_scen_untrained\n",
    "    else:\n",
    "        result[\"train_losses_random_baseline\"] = losses_random\n",
    "        result[\"train_losses_scenario_filter_before_training\"] = losses_scen_untrained\n",
    "        result[\"train_random_baseline_mean\"] = mean_tensor(losses_random)\n",
    "        result[\"train_scenario_filter_before_training_mean\"] = mean_tensor(losses_scen_untrained)\n",
    "\n",
    "        # backward-compatible aliases\n",
    "        result[\"train_losses_before\"] = losses_random\n",
    "        result[\"train_losses_after\"] = losses_scen_untrained\n",
    "\n",
    "    return result"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ba1d2ff6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      node     L         mse       rmse  pinball_avg\n",
      "0    4-2-0  7272   60.860500   7.801314     2.215687\n",
      "1    4-2-1  7272  166.060226  12.886436     3.711401\n",
      "2    4-2-2  7272   96.437782   9.820274     2.793598\n",
      "3    4-2-3  7272  158.961563  12.607996     3.616912\n",
      "4    4-2-4  7272   72.481529   8.513608     2.478071\n",
      "5    4-2-5  7272   83.656120   9.146372     2.681040\n",
      "6    4-2-6  7272   72.326065   8.504473     2.485013\n",
      "7    4-2-7  7272   62.194664   7.886359     2.174087\n",
      "8    4-2-8  7272  103.181175  10.157814     2.997495\n",
      "9    4-2-9  7272   85.795273   9.262574     2.622411\n",
      "10  4-2-10  7272   14.474661   3.804558     1.113066\n"
     ]
    }
   ],
   "source": [
    "window_pack_full_a = sample_window_non_parametric_benchmark(\n",
    "    models_s_non_parametric, handlers_s_non_parametric, pack_data_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    horizon_days=303,\n",
    "    start_day=0,\n",
    "    n_samples=200,\n",
    "    seq_len=24,\n",
    ")\n",
    "\n",
    "dfm_after  = compute_metrics_window(window_pack_full_a)\n",
    "print(dfm_after)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "704a965f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "      node     L         mse       rmse  pinball_avg\n",
      "0    4-2-0  7272   60.869144   7.801868     2.217886\n",
      "1    4-2-1  7272  165.861725  12.878731     3.708451\n",
      "2    4-2-2  7272   96.490944   9.822980     2.791743\n",
      "3    4-2-3  7272  158.983566  12.608869     3.618230\n",
      "4    4-2-4  7272   72.402847   8.508986     2.478066\n",
      "5    4-2-5  7272   83.519562   9.138904     2.679364\n",
      "6    4-2-6  7272   72.419151   8.509944     2.484565\n",
      "7    4-2-7  7272   62.229061   7.888540     2.174851\n",
      "8    4-2-8  7272  103.075089  10.152590     2.998605\n",
      "9    4-2-9  7272   85.890343   9.267704     2.624303\n",
      "10  4-2-10  7272   14.537000   3.812742     1.115312\n"
     ]
    }
   ],
   "source": [
    "models_after = {node: result['dfl_trained'].predictor.models[str(node)] for node in target_nodes}\n",
    "window_pack_full_after = sample_window_non_parametric_benchmark(\n",
    "    models_after, handlers_s_non_parametric, pack_data_s_non_parametric,\n",
    "    target_nodes=target_nodes,\n",
    "    horizon_days=303,\n",
    "    start_day=0,\n",
    "    n_samples=200,\n",
    "    seq_len=24,\n",
    ")\n",
    "\n",
    "dfm_after  = compute_metrics_window(window_pack_full_after)\n",
    "print(dfm_after)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
