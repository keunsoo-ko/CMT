import argparse, os, cv2, glob
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from network.network_pro import Inpaint
from tqdm import tqdm
from utils import *
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Official Pytorch Code for K. Ko and C.-S. Kim, Continuously Masked Transformer for Image Inpainting, ICCV 2023", usage='use "%(prog)s --help" for more information', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--ckpt', required=True, help='Path for the pretrained model')

parser.add_argument('--img_path', default="./samples/test_img", help='''Path for directory of images. Please note that the file name should be same with that of its corresponding mask''')
parser.add_argument('--mask_path', default="./samples/test_mask", help='''Path for directory of masks.''')
parser.add_argument('--output_path', default="./samples/results", help='Path for saving inpainted images')


args = parser.parse_args()

assert os.path.exists(args.img_path), "Please check image path"
assert os.path.exists(args.mask_path), "Please check mask path"

if not os.path.exists(args.output_path):
    os.mkdir(args.output_path)

# Hyper parameters
device = torch.device('cuda')

# Pro
proposed = Inpaint()
proposed = load_checkpoint(args.ckpt, proposed)
proposed.eval().to(device)
maskfn = glob.glob(os.path.join(args.mask_path, '*.*'))
prog_bar = tqdm(maskfn)
avg = 0.
for step, mask_fn in enumerate(prog_bar):
    fn = os.path.basename(mask_fn)
    gt_ = (cv2.imread(os.path.join(args.img_path, fn)) / 255.) * 2 - 1.
    mask = cv2.imread(mask_fn)[..., 0] / 255.
    gt = torch.Tensor(gt_)[None].permute(0, 3, 1, 2).to(device, dtype=torch.float32)
    mask = torch.Tensor(mask)[None, None].to(device, dtype=torch.float32)

    with torch.no_grad():
        out_pro = proposed(gt, mask)
    out_pro = torch.clip(out_pro, -1., 1.)*0.5 + 0.5
    out_pro = out_pro[0].permute(1, 2, 0).cpu().detach().numpy() * 255.
    score = psnr(out_pro, (gt_ * 0.5 + 0.5)*255.)
    save_path_ = os.path.join(args.output_path, '{}').format(fn)
    cv2.imwrite(save_path_, out_pro)
    avg += score
    prog_bar.set_description("PSNR {}".format(avg / (step + 1)))
