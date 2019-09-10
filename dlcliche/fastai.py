"""
Fast.ai 1.0 helper functions
"""

from .image import *


import fastai
import fastai.text
import fastprogress

def fastai_progress_as_text():
    """Disalbe fast.ai progress bar.
    Thanks to https://forums.fast.ai/t/default-to-completely-disable-progress-bar/40010/3
    """
    fastprogress.fastprogress.NO_BAR = True
    master_bar, progress_bar = fastprogress.force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = master_bar, progress_bar
    fastai.basic_data.master_bar, fastai.basic_data.progress_bar = master_bar, progress_bar
    dataclass.master_bar, dataclass.progress_bar = master_bar, progress_bar
    fastai.text.master_bar, fastai.text.progress_bar = master_bar, progress_bar
    fastai.text.data.master_bar, fastai.text.data.progress_bar = master_bar, progress_bar
    fastai.core.master_bar, fastai.core.progress_bar = master_bar, progress_bar


from fastai.callbacks.hooks import hook_output
from fastai.vision import *
from skimage.transform import resize

def visualize_cnn_by_cam(learn, data_index=None, ax=None, ds_type='valid', ds=None, x=None, y=None,
                         label=None, cuda=True, show_original='vertical'):
    """Visualize Grad-CAM of a fast.ai learner.
    Thanks to https://nbviewer.jupyter.org/github/fastai/course-v3/blob/master/nbs/dl1/lesson6-pets-more.ipynb

    Examples:
        visualize_cnn_by_cam(learn, i) # Visualize i-th sample in learn's valid_ds. Class y is ds's value.
        visualize_cnn_by_cam(learn, i, ds=eval_ds) # Visualize i-th sample in external eval_ds.
        visualize_cnn_by_cam(learn, i, ds=eval_ds, y=0) # Ditto, except visualizing class y is always 0.
        visualize_cnn_by_cam(learn, x=raw_tensor.cuda()) # Visualize raw tensor, claass y is predicted.
    """

    def hooked_backward(cat):
        with hook_output(m[0]) as hook_a:
            with hook_output(m[0], grad=True) as hook_g:
                preds = m(xb)
                if cat is None:
                    cat = np.argmax(preds[0])
                preds[0, int(cat)].backward()
        return hook_a, hook_g, cat
    def show_heatmap(x, hm, label, alpha=0.5, ax=None):
        if ax is None: _, ax = plt.subplots(1, 1)
        ax.set_title(label)
        _im = x[0].numpy().transpose(1, 2, 0)
        _cm = resize(plt.cm.magma(plt.Normalize()(hm))[:, :, :3], _im.shape)
        img = (1 - alpha) * _im + alpha * _cm
        if show_original is not None:
            img = np.concatenate([_im, img], axis=0 if show_original == 'vertical' else 1)
        ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.axis('off')
    def cpu(x):
        return x.cpu() if cuda else x

    m = learn.model.eval()

    # Get sample
    if x is None:
        if ds is None:
            ds = learn.data.valid_ds if ds_type.lower() == 'valid' else learn.data.train_ds
        x, _y = ds[data_index]
        if y is None:
            y = _y.data
        if isinstance(y, (list, np.ndarray)): # single label -> one hot encoding
            y = np.where(y)[0]
        xb,_ = learn.data.one_item(x)
        if 'denorm' in dir(learn.data):
            xb_im = Image(learn.data.denorm(xb)[0])
            xb = xb.cuda() if cuda else xb
        else:
            xb_im = Image(xb[0])
        label = str(ds.y[data_index]) if label is None else label
    else:
        if len(x.shape) < 4:
            x.unsqueeze_(0)
        xb, y = x, y
        xb_im = Image(xb[0])

    # Visualize
    hook_a, hook_g, cat = hooked_backward(cat=y)
    if label is None:
        if ds is not None:
            label = ds.classes[cat]
        else:
            label = str(cat)
    acts = cpu(hook_a.stored[0])
    grad = cpu(hook_g.stored[0][0])
    grad_chan = grad.mean(1).mean(1)
    mult = (acts*grad_chan[..., None, None]).mean(0)
    show_heatmap(x=cpu(xb), hm=mult, label=label, ax=ax)