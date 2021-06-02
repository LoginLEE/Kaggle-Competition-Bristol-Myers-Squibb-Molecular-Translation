# coding: utf8
from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
import h5py
import os
import warnings
import shutil


class RawField(object):
    """Defines a general datatype.

    Every dataset consists of one or more types of data. For instance,
    a machine translation dataset contains paired examples of text, while
    an image captioning dataset contains images and texts.
    Each of these types of data is represented by a RawField object.
    An RawField object does not assume any property of the data type and
    it holds parameters relating to how a datatype should be processed.

    Attributes:
        preprocessing: The Pipeline that will be applied to examples
            using this field before creating an example.
            Default: None.
        postprocessing: A Pipeline that will be applied to a list of examples
            using this field before assigning to a batch.
            Function signature: (batch(list)) -> object
            Default: None.
    """

    def __init__(self, preprocessing=None, postprocessing=None):
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing

    def preprocess(self, x):
        """ Preprocess an example if the `preprocessing` Pipeline is provided. """
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, *args, **kwargs):
        """Process a list of examples to create a batch.

        Postprocess the batch with user-provided Pipeline.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            object: Processed object given the input and custom
                postprocessing Pipeline.
        """
        if self.postprocessing is not None:
            batch = self.postprocessing(batch)
        return default_collate(batch)


class Merge(RawField):
    def __init__(self, *fields):
        super(Merge, self).__init__()
        self.fields = fields

    def preprocess(self, x):
        return tuple(f.preprocess(x) for f in self.fields)

    def process(self, batch, *args, **kwargs):
        if len(self.fields) == 1:
            batch = [
                batch,
            ]
        else:
            batch = list(zip(*batch))

        out = list(f.process(b, *args, **kwargs) for f, b in zip(self.fields, batch))
        return out


class ImageDetectionsField(RawField):
    def __init__(
        self,
        preprocessing=None,
        postprocessing=None,
        detections_path=None,
        max_detections=100,
        sort_by_prob=False,
        load_in_tmp=True,
    ):
        self.max_detections = max_detections
        self.detections_path = detections_path
        self.sort_by_prob = sort_by_prob

        tmp_detections_path = os.path.join("/tmp", os.path.basename(detections_path))

        if load_in_tmp:
            if not os.path.isfile(tmp_detections_path):
                if shutil.disk_usage("/tmp")[-1] < os.path.getsize(detections_path):
                    warnings.warn(
                        "Loading from %s, because /tmp has no enough space."
                        % detections_path
                    )
                else:
                    warnings.warn("Copying detection file to /tmp")
                    shutil.copyfile(detections_path, tmp_detections_path)
                    warnings.warn("Done.")
                    self.detections_path = tmp_detections_path
            else:
                self.detections_path = tmp_detections_path

        super(ImageDetectionsField, self).__init__(preprocessing, postprocessing)

    def preprocess(self, x, avoid_precomp=False):
        image_id = int(x.split("_")[-1].split(".")[0])
        try:
            f = h5py.File(self.detections_path, "r")
            precomp_data = f["%d_features" % image_id][()]
            if self.sort_by_prob:
                precomp_data = precomp_data[
                    np.argsort(np.max(f["%d_cls_prob" % image_id][()], -1))[::-1]
                ]
        except KeyError:
            warnings.warn("Could not find detections for %d" % image_id)
            precomp_data = np.random.rand(10, 2048)

        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate(
                [precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0
            )
        elif delta < 0:
            precomp_data = precomp_data[: self.max_detections]

        return precomp_data.astype(np.float32)


class InChIField(RawField):
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,
        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    def __init__(
        self,
        init_token="<sos>",
        eos_token="<eos>",
        fix_length=None,
        dtype=torch.long,
        remove_punctuation=False,
        batch_first=True,
        pad_token="<pad>",
        unk_token="<unk>",
        preprocessing=None,
        postprocessing=None,
        tokenizer_filename="tokenizer.pth",
    ):
        self.init_token = init_token
        self.eos_token = eos_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.remove_punctuation = remove_punctuation
        self.batch_first = batch_first
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.vocab = torch.load(tokenizer_filename)
        print(f"tokenizer.stoi: {self.vocab.stoi}")
        super(InChIField, self).__init__(preprocessing, postprocessing)

    def process(self, batch, device=None):
        padded = self.pad(batch)
        tensor = self.numericalize(padded, device=device)
        return tensor

    def pad(self, minibatch):
        minibatch = list(minibatch)
        max_len = max(len(x) for x in minibatch)  # padding ?
        padded, lengths = [], []
        for x in minibatch:
            padded.append(
                [self.init_token]
                + x[:max_len]
                + [self.eos_token]
                + [self.pad_token] * max(0, max_len - len(x))
            )
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        return padded

    def numericalize(self, arr, device=None):  # turn to number
        arr = self.vocab.texts_to_sequences(arr)
        var = torch.tensor(arr, dtype=self.dtype, device=device)

        if not self.batch_first:
            var.t_()
        var = var.contiguous()

        return var

    def decode(self, word_idxs):  # turn to string
        if isinstance(word_idxs, list) and len(word_idxs) == 0:
            assert 0  # gg
        if isinstance(word_idxs, list) and isinstance(word_idxs[0], int):
            return self.decode([word_idxs])[0]
        elif isinstance(word_idxs, np.ndarray) and word_idxs.ndim == 1:
            return self.decode(word_idxs.reshape((1, -1)))[0]
        elif isinstance(word_idxs, torch.Tensor) and word_idxs.ndimension() == 1:
            return self.decode(word_idxs.unsqueeze(0))[0]

        return self.vocab.predict_captions(word_idxs)
