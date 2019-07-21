import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import numpy.linalg as npl

import imagelib
from facelib import LandmarksProcessor
from imagelib import IEPolys
from interact import interact as io
from utils import Path_utils
from utils.cv2_utils import *
from utils.DFLJPG import DFLJPG
from utils.DFLPNG import DFLPNG

class MaskEditor:
    STATE_NONE=0
    STATE_MASKING=1

    def __init__(self, img, prev_images, next_images, mask=None, ie_polys=None, get_status_lines_func=None):
        self.img = imagelib.normalize_channels (img,3)
        h, w, c = img.shape

        if h != w and w != 256:
            #to support any square res, scale img,mask and ie_polys to 256, then scale ie_polys back on .get_ie_polys()
            raise Exception ("MaskEditor does not support image size != 256x256")

        ph, pw = h // 4, w // 4 #pad wh

        self.prev_images = prev_images
        self.next_images = next_images

        if mask is not None:
            self.mask = imagelib.normalize_channels (mask,3)
        else:
            self.mask = np.zeros ( (h,w,3) )
        self.get_status_lines_func = get_status_lines_func

        self.state_prop = self.STATE_NONE

        self.w, self.h = w, h
        self.pw, self.ph = pw, ph
        self.pwh = np.array([self.pw, self.ph])
        self.pwh2 = np.array([self.pw*2, self.ph*2])
        self.sw, self.sh = w+pw*2, h+ph*2
        self.prwh = 64 #preview wh

        if ie_polys is None:
            ie_polys = IEPolys()
        self.ie_polys = ie_polys

        self.polys_mask = None
        self.preview_images = None

        self.mouse_x = self.mouse_y = 9999
        self.screen_status_block = None
        self.screen_status_block_dirty = True
        self.screen_changed = True

    def set_state(self, state):
        self.state = state

    @property
    def state(self):
        return self.state_prop

    @state.setter
    def state(self, value):
        self.state_prop = value
        if value == self.STATE_MASKING:
            self.ie_polys.dirty = True

    def get_mask(self):
        if self.ie_polys.switch_dirty():
            self.screen_status_block_dirty = True
            self.ie_mask = img = self.mask.copy()

            self.ie_polys.overlay_mask(img)

            return img
        return self.ie_mask

    def get_screen_overlay(self):
        img = np.zeros ( (self.sh, self.sw, 3) )

        if self.state == self.STATE_MASKING:
            mouse_xy = self.mouse_xy.copy() + self.pwh
            l = self.ie_polys.n_list()
            if l.n > 0:
                p = l.cur_point().copy() + self.pwh
                color = (0,1,0) if l.type == 1 else (0,0,1)
                cv2.line(img, tuple(p), tuple(mouse_xy), color )

        return img

    def undo_to_begin_point(self):
        while not self.undo_point():
            pass

    def undo_point(self):
        self.screen_changed = True
        if self.state == self.STATE_NONE:
            if self.ie_polys.n > 0:
                self.state = self.STATE_MASKING

        if self.state == self.STATE_MASKING:
            if self.ie_polys.n_list().n_dec() == 0 and \
               self.ie_polys.n_dec() == 0:
                self.state = self.STATE_NONE
            else:
                return False

        return True

    def redo_to_end_point(self):
        while not self.redo_point():
            pass

    def redo_point(self):
        self.screen_changed = True
        if self.state == self.STATE_NONE:
            if self.ie_polys.n_max > 0:
                self.state = self.STATE_MASKING
                if self.ie_polys.n == 0:
                    self.ie_polys.n_inc()

        if self.state == self.STATE_MASKING:
            while True:
                l = self.ie_polys.n_list()
                if l.n_inc() == l.n_max:
                    if self.ie_polys.n == self.ie_polys.n_max:
                        break
                    self.ie_polys.n_inc()
                else:
                    return False

        return True

    def combine_screens(self, screens):

        screens_len = len(screens)

        new_screens = []
        for screen, padded_overlay in screens:
            screen_img = np.zeros( (self.sh, self.sw, 3), dtype=np.float32 )

            screen = imagelib.normalize_channels (screen, 3)
            h,w,c = screen.shape

            screen_img[self.ph:-self.ph, self.pw:-self.pw, :] = screen

            if padded_overlay is not None:
                screen_img = screen_img + padded_overlay

            screen_img = np.clip(screen_img*255, 0, 255).astype(np.uint8)
            new_screens.append(screen_img)

        return np.concatenate (new_screens, axis=1)

    def get_screen_status_block(self, w, c):
        if self.screen_status_block_dirty:
            self.screen_status_block_dirty = False
            lines = [
                    'Polys current/max = %d/%d' % (self.ie_polys.n, self.ie_polys.n_max),
                    ]
            if self.get_status_lines_func is not None:
                lines += self.get_status_lines_func()

            lines_count = len(lines)


            h_line = 21
            h = lines_count * h_line
            img = np.ones ( (h,w,c) ) * 0.1

            for i in range(lines_count):
                img[ i*h_line:(i+1)*h_line, 0:w] += \
                    imagelib.get_text_image (  (h_line,w,c), lines[i], color=[0.8]*c )

            self.screen_status_block = np.clip(img*255, 0, 255).astype(np.uint8)

        return self.screen_status_block

    def set_screen_status_block_dirty(self):
        self.screen_status_block_dirty = True

    def set_screen_changed(self):
        self.screen_changed = True

    def switch_screen_changed(self):
        result = self.screen_changed
        self.screen_changed = False
        return result

    def make_screen(self):
        screen_overlay = self.get_screen_overlay()
        final_mask = self.get_mask()

        masked_img = self.img*final_mask*0.5 + self.img*(1-final_mask)

        pink = np.full ( (self.h, self.w, 3), (1,0,1) )
        pink_masked_img = self.img*final_mask + pink*(1-final_mask)




        screens = [ (self.img, screen_overlay),
                    (masked_img, screen_overlay),
                    (pink_masked_img, screen_overlay),
                    ]
        screens = self.combine_screens(screens)

        if self.preview_images is None:
            sh,sw,sc = screens.shape

            prh, prw = self.prwh, self.prwh

            total_w = sum ([ img.shape[1] for (t,img) in self.prev_images ]) + \
                      sum ([ img.shape[1] for (t,img) in self.next_images ])

            total_images_len = len(self.prev_images) + len(self.next_images)

            max_hor_images_count = sw // prw
            max_side_images_count = (max_hor_images_count - 1) // 2

            prev_images = self.prev_images[-max_side_images_count:]
            next_images = self.next_images[:max_side_images_count]

            border = 2

            max_wh_bordered = (prw-border*2, prh-border*2)

            prev_images = [ (t, cv2.resize( imagelib.normalize_channels(img, 3), max_wh_bordered )) for t,img in prev_images ]
            next_images = [ (t, cv2.resize( imagelib.normalize_channels(img, 3), max_wh_bordered )) for t,img in next_images ]

            for images in [prev_images, next_images]:
                for i, (t, img) in enumerate(images):
                    new_img = np.zeros ( (prh,prw, sc) )
                    new_img[border:-border,border:-border] = img

                    if t == 2:
                        cv2.line (new_img, (       prw//2, int(prh//1.5) ), (int(prw/1.5), prh      ) , (0,1,0), thickness=2 )
                        cv2.line (new_img, ( int(prw/1.5), prh           ), (         prw, prh // 2 ) , (0,1,0), thickness=2 )
                    elif t == 1:
                        cv2.line (new_img, ( prw//2, prh//2 ), ( prw, prh      ) , (0,0,1), thickness=2 )
                        cv2.line (new_img, ( prw//2, prh    ), ( prw, prh // 2 ) , (0,0,1), thickness=2 )

                    images[i] = new_img


            preview_images = []
            if len(prev_images) > 0:
                preview_images += [ np.concatenate (prev_images, axis=1) ]

            img = np.full ( (prh,prw, sc), (0,0,1), dtype=np.float )
            img[border:-border,border:-border] = cv2.resize( self.img, max_wh_bordered )

            preview_images += [ img ]

            if len(next_images) > 0:
                preview_images += [ np.concatenate (next_images, axis=1) ]

            preview_images = np.concatenate ( preview_images, axis=1  )

            left_pad = sw // 2 - len(prev_images) * prw - prw // 2
            right_pad = sw // 2 - len(next_images) * prw - prw // 2

            preview_images = np.concatenate ([np.zeros ( (preview_images.shape[0], left_pad, preview_images.shape[2]) ),
                                              preview_images,
                                              np.zeros ( (preview_images.shape[0], right_pad, preview_images.shape[2]) )
                                             ], axis=1)
            self.preview_images = np.clip(preview_images * 255, 0, 255 ).astype(np.uint8)

        status_img = self.get_screen_status_block( screens.shape[1], screens.shape[2] )

        result = np.concatenate ( [self.preview_images, screens, status_img], axis=0  )

        return result

    def mask_finish(self, n_clip=True):
        if self.state == self.STATE_MASKING:
            self.screen_changed = True
            if self.ie_polys.n_list().n <= 2:
                self.ie_polys.n_dec()
            self.state = self.STATE_NONE
            if n_clip:
                self.ie_polys.n_clip()

    def set_mouse_pos(self,x,y):
        if self.preview_images is not None:
            y -= self.preview_images.shape[0]

        mouse_x = x % (self.sw) - self.pw
        mouse_y = y % (self.sh) - self.ph



        if mouse_x != self.mouse_x or mouse_y != self.mouse_y:
            self.mouse_xy = np.array( [mouse_x, mouse_y] )
            self.mouse_x, self.mouse_y = self.mouse_xy
            self.screen_changed = True

    def mask_point(self, type):
        self.screen_changed = True
        if self.state == self.STATE_MASKING and \
           self.ie_polys.n_list().type != type:
            self.mask_finish()

        elif self.state == self.STATE_NONE:
            self.state = self.STATE_MASKING
            self.ie_polys.add(type)

        if self.state == self.STATE_MASKING:
            self.ie_polys.n_list().add (self.mouse_x, self.mouse_y)

    def get_ie_polys(self):
        return self.ie_polys

def mask_editor_main(input_dir, confirmed_dir=None, skipped_dir=None):
    input_path = Path(input_dir)

    confirmed_path = Path(confirmed_dir)
    skipped_path = Path(skipped_dir)

    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    if not confirmed_path.exists():
        confirmed_path.mkdir(parents=True)

    if not skipped_path.exists():
        skipped_path.mkdir(parents=True)

    wnd_name = "MaskEditor tool"
    io.named_window (wnd_name)
    io.capture_mouse(wnd_name)
    io.capture_keys(wnd_name)

    cached_images = {}

    image_paths = [ Path(x) for x in Path_utils.get_image_paths(input_path)]
    done_paths = []
    done_images_types = {}
    image_paths_total = len(image_paths)

    zoom_factor = 1.0
    preview_images_count = 9
    target_wh = 256

    do_prev_count = 0
    do_save_move_count = 0
    do_save_count = 0
    do_skip_move_count = 0
    do_skip_count = 0

    def jobs_count():
        return do_prev_count + do_save_move_count + do_save_count + do_skip_move_count + do_skip_count

    is_exit = False
    while not is_exit:

        if len(image_paths) > 0:
            filepath = image_paths.pop(0)
        else:
            filepath = None

        next_image_paths = image_paths[0:preview_images_count]
        next_image_paths_names = [ path.name for path in next_image_paths ]
        prev_image_paths = done_paths[-preview_images_count:]
        prev_image_paths_names = [ path.name for path in prev_image_paths ]

        for key in list( cached_images.keys() ):
            if key not in prev_image_paths_names and \
               key not in next_image_paths_names:
                cached_images.pop(key)

        for paths in [prev_image_paths, next_image_paths]:
            for path in paths:
                if path.name not in cached_images:
                    cached_images[path.name] = cv2_imread(str(path)) / 255.0

        if filepath is not None:
            if filepath.suffix == '.png':
                dflimg = DFLPNG.load( str(filepath) )
            elif filepath.suffix == '.jpg':
                dflimg = DFLJPG.load ( str(filepath) )
            else:
                dflimg = None

            if dflimg is None:
                io.log_err ("%s is not a dfl image file" % (filepath.name) )
                continue
            else:
                lmrks = dflimg.get_landmarks()
                ie_polys = dflimg.get_ie_polys()
                fanseg_mask = dflimg.get_fanseg_mask()

                if filepath.name in cached_images:
                    img = cached_images[filepath.name]
                else:
                    img = cached_images[filepath.name] = cv2_imread(str(filepath)) / 255.0

                if fanseg_mask is not None:
                    mask = fanseg_mask
                else:
                    mask = LandmarksProcessor.get_image_hull_mask( img.shape, lmrks)
        else:
            img = np.zeros ( (target_wh,target_wh,3) )
            mask = np.ones ( (target_wh,target_wh,3) )
            ie_polys = None

        def get_status_lines_func():
            return ['Progress: %d / %d . Current file: %s' % (len(done_paths), image_paths_total, str(filepath.name) if filepath is not None else "end" ),
                    '[Left mouse button] - mark include mask.',
                    '[Right mouse button] - mark exclude mask.',
                    '[Middle mouse button] - finish current poly.',
                    '[Mouse wheel] - undo/redo poly or point. [+ctrl] - undo to begin/redo to end',
                    '[q] - prev image. [w] - skip and move to %s. [e] - save and move to %s. ' % (skipped_path.name, confirmed_path.name),
                    '[z] - prev image. [x] - skip. [c] - save. ',
                    'hold [shift] - speed up the frame counter by 10.',
                    '[-/+] - window zoom [esc] - quit',
                    ]

        try:
            ed = MaskEditor(img,
                            [ (done_images_types[name], cached_images[name]) for name in prev_image_paths_names ],
                            [ (0, cached_images[name]) for name in next_image_paths_names ],
                            mask, ie_polys, get_status_lines_func)
        except Exception as e:
            print(e)
            continue

        next = False
        while not next:
            io.process_messages(0.005)

            if jobs_count() == 0:
                for (x,y,ev,flags) in io.get_mouse_events(wnd_name):
                    x, y = int (x / zoom_factor), int(y / zoom_factor)
                    ed.set_mouse_pos(x, y)
                    if filepath is not None:
                        if ev == io.EVENT_LBUTTONDOWN:
                            ed.mask_point(1)
                        elif ev == io.EVENT_RBUTTONDOWN:
                            ed.mask_point(0)
                        elif ev == io.EVENT_MBUTTONDOWN:
                            ed.mask_finish()
                        elif ev == io.EVENT_MOUSEWHEEL:
                            if flags & 0x80000000 != 0:
                                if flags & 0x8 != 0:
                                    ed.undo_to_begin_point()
                                else:
                                    ed.undo_point()
                            else:
                                if flags & 0x8 != 0:
                                    ed.redo_to_end_point()
                                else:
                                    ed.redo_point()

                for key, chr_key, ctrl_pressed, alt_pressed, shift_pressed in io.get_key_events(wnd_name):
                    if chr_key == 'q' or chr_key == 'z':
                        do_prev_count = 1 if not shift_pressed else 10
                    elif chr_key == '-':
                        zoom_factor = np.clip (zoom_factor-0.1, 0.1, 4.0)
                        ed.set_screen_changed()
                    elif chr_key == '+':
                        zoom_factor = np.clip (zoom_factor+0.1, 0.1, 4.0)
                        ed.set_screen_changed()
                    elif key == 27: #esc
                        is_exit = True
                        next = True
                        break
                    elif filepath is not None:
                        if chr_key == 'e':
                            do_save_move_count = 1 if not shift_pressed else 10
                        elif chr_key == 'c':
                            do_save_count = 1 if not shift_pressed else 10
                        elif chr_key == 'w':
                            do_skip_move_count = 1 if not shift_pressed else 10
                        elif chr_key == 'x':
                            do_skip_count = 1 if not shift_pressed else 10

            if do_prev_count > 0:
                do_prev_count -= 1
                if len(done_paths) > 0:
                    if filepath is not None:
                        image_paths.insert(0, filepath)

                    filepath = done_paths.pop(-1)
                    done_images_types[filepath.name] = 0

                    if filepath.parent != input_path:
                        new_filename_path = input_path / filepath.name
                        filepath.rename ( new_filename_path )
                        image_paths.insert(0, new_filename_path)
                    else:
                        image_paths.insert(0, filepath)

                    next = True
            elif filepath is not None:
                if do_save_move_count > 0:
                    do_save_move_count -= 1

                    ed.mask_finish()
                    dflimg.embed_and_set (str(filepath), ie_polys=ed.get_ie_polys() )

                    done_paths += [ confirmed_path / filepath.name ]
                    done_images_types[filepath.name] = 2
                    filepath.rename(done_paths[-1])

                    next = True
                elif do_save_count > 0:
                    do_save_count -= 1

                    ed.mask_finish()
                    dflimg.embed_and_set (str(filepath), ie_polys=ed.get_ie_polys() )

                    done_paths += [ filepath ]
                    done_images_types[filepath.name] = 2

                    next = True
                elif do_skip_move_count > 0:
                    do_skip_move_count -= 1

                    done_paths += [ skipped_path / filepath.name ]
                    done_images_types[filepath.name] = 1
                    filepath.rename(done_paths[-1])

                    next = True
                elif do_skip_count > 0:
                    do_skip_count -= 1

                    done_paths += [ filepath ]
                    done_images_types[filepath.name] = 1

                    next = True
            else:
                do_save_move_count = do_save_count = do_skip_move_count = do_skip_count = 0

            if jobs_count() == 0:
                if ed.switch_screen_changed():
                    screen = ed.make_screen()
                    if zoom_factor != 1.0:
                        h,w,c = screen.shape
                        screen = cv2.resize ( screen, ( int(w*zoom_factor), int(h*zoom_factor) ) )
                    io.show_image (wnd_name, screen )


        io.process_messages(0.005)

    io.destroy_all_windows()

