# 在50系卡上安装AnySplat环境

## 创建环境

! 修改`requirements.txt`文件，将`xformers==0.0.24`改为`xformers==0.0.30`

```
uv venv --python 3.10

source .venv/bin/activate

<!-- uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 -->

uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

uv pip install -r requirements.txt
```


## 解决`Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead`的问题：
```
cd src/model/encoder/backbone/croco/curope/
```
把 kernels.cu中的

```
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.type(), "rope_2d_cuda", ([&] {
改为
AT_DISPATCH_FLOATING_TYPES_AND_HALF(tokens.scalar_type(), "rope_2d_cuda", ([&] {
```
然后执行
```
python setup.py build_ext --inplace
```

