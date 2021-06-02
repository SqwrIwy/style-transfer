import json
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
from flask import send_from_directory

cnt = 0

alpha = None
pc = None

UPLOAD_FOLDER = '/uploads'  #文件存放路径
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png']) #限制上传文件格式
  
app = Flask(__name__)#实例化app对象

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  
testInfo = {}
  
@app.route('/index')
def index():
    global cnt
    cnt += 1
    with open('templates/index.html', 'r') as f:
        a = f.read()
        a = a.replace('UNIQUE_ID', str(cnt))
        with open('templates/index_{}.html'.format(cnt), 'w') as g:
            g.write(a)
    return render_template('index_{}.html'.format(cnt))

@app.route('/upload/content/<path:path>', methods=['GET', 'POST'])
def upload_content(path):
    if request.method == 'POST':
        file = request.files['image']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            l = glob.glob('uploads/content_' + path + '*')
            if len(l) > 0:
                os.system('rm {}'.format(l[0]))
            file.save(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'][1:], 'content_' + path + filename[filename.rfind('.'):]))
            return  '{"filename":"%s"}' % filename
    return ''

@app.route('/upload/style/<path:path>', methods=['GET', 'POST'])
def upload_style(path):
    if request.method == 'POST':
        file = request.files['image']

        if file and allowed_file(file.filename):
            l = glob.glob('uploads/style_' + path + '*')
            if len(l) > 0:
                os.system('rm {}'.format(l[0]))
            filename = secure_filename(file.filename)
            file.save(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'][1:], 'style_' + path + filename[filename.rfind('.'):]))
            return  '{"filename":"%s"}' % filename
    return ''


import glob
from pathlib import Path
import cv2

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

def style_transfer_mask(vgg, decoder, content, mask, style1, style2, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style1_f = vgg(style1)
    style2_f = vgg(style2)
    feat1 = adaptive_instance_normalization(content_f, style1_f)
    feat2 = adaptive_instance_normalization(content_f, style2_f)
    H, W = feat1.size()[-2:]
    tr = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
    ])
    mask = tr(mask)
    mask = mask.view(1, 1, H, W)
    feat = feat1 * mask.expand_as(feat1) + feat2 - feat2 * mask.expand_as(feat2)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


class Args:
    def __init__(self, t):
        self.content = 'uploads/content_{}.jpg'.format(t)
        self.style = 'uploads/style_{}.jpg'.format(t)
        self.vgg = 'models/vgg_normalised.pth'
        self.decoder = 'models/decoder.pth'
        self.content_size = 0
        self.style_size = 0
        self.save_ext = '.jpg'
        self.output = 'output'
        self.crop = None
        self.preserve_color = None
        self.alpha = 1.0
        self.style_interpolation_weights = ''
        self.mask = ''
        self.style2 = ''

def work(path):

    args = Args(path)

    global alpha, pc

    args.alpha = alpha

    if pc == 'true':
        args.preserve_color = True
    else:
        args.preserve_color = False

    do_interpolation = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Either --content or --contentDir should be given.
    assert (args.content or args.content_dir)
    if args.content:
        content_paths = [Path(args.content)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --styleDir should be given.
    assert (args.style or args.style_dir)
    if args.style:
        style_paths = args.style.split(',')
        if len(style_paths) == 1:
            style_paths = [Path(args.style)]
        else:
            do_interpolation = True
            assert (args.style_interpolation_weights != ''), \
                'Please specify interpolation weights'
            weights = [int(i) for i in args.style_interpolation_weights.split(',')]
            interpolation_weights = [w / sum(weights) for w in weights]
    else:
        style_dir = Path(args.style_dir)
        style_paths = [f for f in style_dir.glob('*')]

    decoder = net.decoder
    vgg = net.vgg

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load(args.decoder))
    vgg.load_state_dict(torch.load(args.vgg))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(args.content_size, args.crop)
    style_tf = test_transform(args.style_size, args.crop)

    for content_path in content_paths:
        if do_interpolation:  # one content image, N style image
            style = torch.stack([style_tf(Image.open(str(p))) for p in style_paths])
            content = content_tf(Image.open(str(content_path))) \
                .unsqueeze(0).expand_as(style)
            style = style.to(device)
            content = content.to(device)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style,
                                        args.alpha, interpolation_weights)
            output = output.cpu()
            output_name = 'static/result_{}.jpg'.format(path)
            save_image(output, str(output_name))

        else:  # process one content and one style
            if args.mask == '':
                for style_path in style_paths:
                    content = content_tf(Image.open(str(content_path)))
                    style = style_tf(Image.open(str(style_path)))
                    if args.preserve_color:
                        style = coral(style, content)
                    style = style.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)
                    with torch.no_grad():
                        output = style_transfer(vgg, decoder, content, style,
                                                args.alpha)
                    output = output.cpu()

                    output_name = 'static/result_{}.jpg'.format(path)
                    save_image(output, str(output_name))
            else:
                for style_path in style_paths:
                    content = content_tf(Image.open(str(content_path)))
                    style = style_tf(Image.open(str(style_path)))
                    style2 = style_tf(Image.open(args.style2))
                    mask = cv2.imread(args.mask)
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                    mask = mask / 255
                    mask = style_tf(Image.fromarray(mask))
                    if args.preserve_color:
                        style = coral(style, content)
                        style2 = coral(style2, content)
                    style = style.to(device).unsqueeze(0)
                    style2 = style2.to(device).unsqueeze(0)
                    content = content.to(device).unsqueeze(0)
                    with torch.no_grad():
                        output = style_transfer_mask(vgg, decoder, content, mask, style, style2, 
                                                args.alpha)
                    output = output.cpu()

                    output_name = 'static/result_{}.jpg'.format(path)
                    save_image(output, str(output_name))





@app.route('/process/<path:path>')
def process(path):
    global alpha, pc
    for i in request.args:
        print(i)
        t = i.find(' ')
        alpha = eval(i[:t]) / 1000
        pc = i[t + 1:]
    path1 = glob.glob(os.getcwd() + '/uploads/content_{}.*'.format(path))[0]
    path2 = glob.glob(os.getcwd() + '/uploads/style_{}.*'.format(path))[0]
    img = cv2.imread(path1)
    os.system('rm {}'.format(path1))
    cv2.imwrite('uploads/content_{}.jpg'.format(path), img)
    img = cv2.imread(path2)
    os.system('rm {}'.format(path2))
    cv2.imwrite('uploads/style_{}.jpg'.format(path), img)
    work(path)
    img = cv2.imread('static/result_{}.jpg'.format(path))
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    cv2.imwrite('static/result_{}.jpg'.format(path), img)
    H = img.shape[0]
    W = img.shape[1]
    if H < W:
        img = cv2.resize(img, (512, 512 * H // W))
    else:
        img = cv2.resize(img, (512 * W // H, 512))
    cv2.imwrite('static/result_{}_512.jpg'.format(path), img)
    return ''


if __name__ == '__main__':
    os.system('rm uploads/*')
    os.system('rm static/*')
    app.run(host='0.0.0.0',  # 任何ip都可以访问
          port=7777,  # 端口
          debug=True
          )
