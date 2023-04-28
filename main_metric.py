import os
import random
import shutil

import cv2
import lpips
from PIL import Image
from tqdm import tqdm
import numpy as np
import argparse
import torch.utils.data
import torchvision.transforms as transforms
from trainer import Trainer
from utils import get_config, unloader, get_model_list

from fid import calculate_fid_given_paths

def fid(real, fake, gpu, batch_size=50, dims=2048):
    print('Calculating FID...')
    print('real dir: {}'.format(real))
    print('fake dir: {}'.format(fake))
    # command = 'python -m pytorch_fid {} {} --gpu {}'.format(real, fake, gpu)  # pytorch-fid 0.1.1
    #command = 'python -m pytorch_fid {} {} --device cuda:{}'.format(real, fake, gpu)  # pytorch-fid 0.2.1
    # command = 'python -m pytorch_fid {} {}'.format(real, fake)
    #os.system(command)

    device = torch.device(gpu)
    fid_score = calculate_fid_given_paths((real, fake), batch_size=batch_size, device=device, dims=dims)
    return fid_score


def LPIPS(root):
    print('Calculating LPIPS...')
    loss_fn_vgg = lpips.LPIPS(net='vgg')
    model = loss_fn_vgg
    model.cuda()

    files = os.listdir(root)
    data = {}
    for file in tqdm(files, desc='loading data'):
        cls = file.split('_')[0]
        idx = int(file.split('_')[1][:-4])
        img = lpips.im2tensor(cv2.resize(lpips.load_image(os.path.join(root, file)), (32, 32)))
        data.setdefault(cls, {})[idx] = img

    classes = set([file.split('_')[0] for file in files])
    res = []
    for cls in tqdm(classes):
        temp = []
        files_cls = [file for file in files if file.startswith(cls + '_')]
        for i in range(0, len(files_cls) - 1, 1):
            # print(i, end='\r')
            for j in range(i + 1, len(files_cls), 1):
                img1 = data[cls][i].cuda()
                img2 = data[cls][j].cuda()

                d = model(img1, img2, normalize=True)
                temp.append(d.detach().cpu().numpy())
        res.append(np.mean(temp))
    lpips_score = np.mean(res)
    print(lpips_score)
    return lpips_score



def eval_scores(data, n_cond, trainer, real_dir, fake_dir, transform):
    if os.path.exists(fake_dir):
        shutil.rmtree(fake_dir)
    os.makedirs(fake_dir, exist_ok=True)
    if os.path.exists(real_dir):
        shutil.rmtree(real_dir)
    os.makedirs(real_dir, exist_ok=True)

    per = np.random.permutation(data.shape[1])
    data = data[:, per, :, :, :]

    num = n_cond
    data_for_gen = data[:, :num, :, :, :]
    data_for_fid = data[:, num:num+128, :, :, :]
    if os.path.exists(real_dir):
        for cls in tqdm(range(data_for_fid.shape[0]), desc='preparing real images'):
            for i in range(128):
                if data_for_fid.shape[1] < 128:
                    idx = np.random.choice(data_for_fid.shape[1], 1).item()
                else:
                    idx = i
                real_img = data_for_fid[cls, idx, :, :, :]
                if args.dataset == 'vggface':
                    real_img *= 255
                real_img = Image.fromarray(np.uint8(real_img))
                real_img.save(os.path.join(real_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    if os.path.exists(fake_dir):
        for cls in tqdm(range(data_for_gen.shape[0]), desc='generating fake images'):
            for i in range(128):
                idx = np.random.choice(data_for_gen.shape[1], args.n_sample_test)
                imgs = data_for_gen[cls, idx, :, :, :]
                imgs = torch.cat([transform(img).unsqueeze(0) for img in imgs], dim=0).unsqueeze(0).cuda()
                fake_x = trainer.generate(imgs)
                output = unloader(fake_x[0].cpu())
                output.save(os.path.join(fake_dir, '{}_{}.png'.format(cls, str(i).zfill(3))), 'png')

    fid_score=fid(real_dir, fake_dir, int(args.gpu))
    lpips_score=LPIPS(fake_dir)

    shutil.rmtree(real_dir)
    shutil.rmtree(fake_dir)
    return fid_score, lpips_score


parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--real_dir', type=str)
parser.add_argument('--fake_dir', type=str)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--n_sample_test', type=int, default=3)
parser.add_argument('--n_test', type=int, default=3)
parser.add_argument('--n_cond', type=int, default=10)
parser.add_argument('--n_exps', type=int, default=3)
parser.add_argument('--eval_path', type=str, default="eval_results.txt")
parser.add_argument('--use_modified_datasets', action='store_true')
args = parser.parse_args()

conf_file = os.path.join(args.name, 'configs.yaml')
config = get_config(conf_file)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
transform = transforms.Compose(transform_list)


if __name__ == '__main__':
    # SEED = 0
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    real_dir = args.real_dir
    fake_dir = os.path.join(args.name, args.fake_dir)
    print('real dir: ', real_dir)
    print('fake dir: ', fake_dir)


    data = np.load(config['data_root'])
    if args.dataset == 'flower':
        if args.use_modified_datasets:
            data = np.load("datasets/flowers2.npy")
        else:
            data = data[85:]
    elif args.dataset == 'animal':
        if args.use_modified_datasets:
            data = np.load("datasets/animal2.npy")
        else:
            data = data[119:]
    elif args.dataset == 'vggface':
        if args.use_modified_datasets:
            data = np.load("datasets/vggface2.npy")
        else:
            data = data[1802:]

    trainer = Trainer(config)
    if args.ckpt:
        last_model_name = os.path.join(args.name, 'checkpoints', args.ckpt)
    else:
        last_model_name = get_model_list(os.path.join(args.name, 'checkpoints'), "gen")
    trainer.load_ckpt(last_model_name)
    trainer.cuda()
    trainer.eval()
    

    fid_scores = []
    lpips_scores=[]
    for i in range(args.n_exps):
        fid_score, lpips_score = eval_scores(data, args.n_cond, trainer, real_dir, fake_dir, transform)
        fid_scores.append(fid_score)
        lpips_scores.append(lpips_score)
    fid_out = sum(fid_scores) / len(fid_scores)
    lpips_out = sum(lpips_scores) / len(lpips_scores)

    with open(args.eval_path, 'a') as writer:
        fid_scores_str = ", ".join(["%.2f" % (x,) for x in fid_scores])
        lpips_scores_str = ", ".join(["%.4f" % (x,) for x in lpips_scores])
        writer.write("%s:\tFID: %.2f (%s)\tLPIPS: %.4f (%s)\n" % (args.dataset+"_"+str(args.n_cond), fid_out, fid_scores_str, lpips_out, lpips_scores_str)) 
