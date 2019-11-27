#-*- coding:utf-8 -*-
import os
import sys
import time
import random
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#图像拼接 参数 灰度图 原图grouth true 预测的图
#paste函数 将图片的一部分用另一张图替换
def stitch_images(grayscale, original, pred):
    gap = 5
    width, height = original[0][:, :, 0].shape
    img_per_row = 2 if width > 200 else 4
    img = Image.new('RGB', (width * img_per_row * 3 + gap * (img_per_row - 1), height * int(len(original) / img_per_row)))

    grayscale = np.array(grayscale).squeeze()
    original = np.array(original)
    pred = np.array(pred)

    for ix in range(len(original)):
        xoffset = int(ix % img_per_row) * width * 3 + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height
        im1 = Image.fromarray(grayscale[ix])
        im2 = Image.fromarray(original[ix])
        im3 = Image.fromarray((pred[ix] * 255).astype(np.uint8))
        img.paste(im1, (xoffset, yoffset))
        img.paste(im2, (xoffset + width, yoffset))
        img.paste(im3, (xoffset + width + width, yoffset))

    return img

##创建文件夹
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir


#解压文件
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#平均移动
#moving average:a succession of averages derived from successive segments (typically of constant size and overlapping) of a series of values.
#从一系列值的连续段（通常具有恒定的大小和重叠）得出的一系列平均值。从一系列值的连续段（通常具有恒定的大小和重叠）得出的一系列平均值。
def moving_average(data, window_width):
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    return ma_vec

##展示一张标题为str 的图
def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()

#保存图片
def imsave(img, path):
    im = Image.fromarray(np.array(img).astype(np.uint8).squeeze())
    im.save(path)

##图灵测试 返回值 0 或者 1 用于打分
#numpy.random.binomial(n,p,size=None) 随机得到一个满足这个二项分布的值 
#鼠标响应事件 onclick事件 plt.gcf().canvas.mpl_connect
#plt.gcf().canvas.start_event_loop(delay) 开始事件循环 
#plt.gcf().canvas.stop_event_loop() 结束事件循环
def turing_test(real_img, fake_img, delay=0):
    height, width, _ = real_img.shape
    imgs = np.array([real_img, (fake_img * 255).astype(np.uint8)])
    real_index = np.random.binomial(1, 0.5)
    fake_index = (real_index + 1) % 2

    img = Image.new('RGB', (2 + width * 2, height))
    img.paste(Image.fromarray(imgs[real_index]), (0, 0))
    img.paste(Image.fromarray(imgs[fake_index]), (2 + width, 0))

    img.success = 0

    def onclick(event):
        if event.xdata is not None:
            if event.x < width and real_index == 0:
                img.success = 1

            elif event.x > width and real_index == 1:
                img.success = 1

        plt.gcf().canvas.stop_event_loop()

    plt.ion()
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    plt.title('click on the real image')
    plt.axis('off')
    plt.imshow(img, interpolation='none')
    plt.show()
    plt.draw()
    plt.gcf().canvas.start_event_loop(delay)

    return img.success

#可视化
def visualize(train_log_file, test_log_file, window_width, title=''):
    train_data = np.loadtxt(train_log_file)
    test_data = np.loadtxt(test_log_file)

    if len(train_data.shape) < 2:
        return

    if len(train_data) < window_width:
        window_width = len(train_data) - 1

    fig = plt.gcf()
    fig.canvas.set_window_title(title)

    plt.ion()
    plt.subplot('121')
    plt.cla()
    if len(train_data) > 1:
        plt.plot(moving_average(train_data[:, 8], window_width))
    plt.title('train')

    plt.subplot('122')
    plt.cla()
    if len(test_data) > 1:
        plt.plot(test_data[:, 8])
    plt.title('test')

    plt.show()
    plt.draw()
    plt.pause(.01)


##展示进度条
##进度条工具
##“”“显示进度条。
#
#     参数：
#         target：预期的步骤总数，如果未知，则为“无”。
#         width：进度条在屏幕上的宽度。
#         verbose：详细模式，0（静音），1（详细），2（半详细）
#         stateful_metrics：可迭代的指标的字符串名称
#             *不应*随时间推移取平均值。 此列表中的指标
#             将按原样显示。 所有其他都将被平均
#             在显示之前按一下进度条。
#         interval：最小视觉进度更新间隔（以秒为单位）。
#     “”

#“”“更新进度栏。
#
#         参数：
#             current：当前步骤的索引。
#             values：元组列表：
#                 `(name, value_for_last_step)`。
#                 如果“name”在“ stateful_metrics”中，
#                 “ value_for_last_step”将按原样显示。
#                 否则，将显示该指标随时间的平均值。
#         “”
class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
