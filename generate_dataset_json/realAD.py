import os
import json
from pathlib import Path


class RealADSolver(object):
    """
    RealAD数据集meta.json生成器

    """
    CLSNAMES = [
       'audiojack', 'pcb', 'phone_battery', 'sim_card_set', 'switch',
                    'terminalblock', 'toothbrush', 'bottle_cap', 'end_cap', 'fire_hood',
                    'mounts', 'plastic_nut', 'plastic_plug', 'regulator', 'rolled_strip_base',
                    'tape', 'porcelain_doll', 'mint', 'eraser', 'button_battery',
                    'toy', 'transistor1', 'usb',
    'usb_adaptor', 'zipper',  'toy_brick', 'u_block', 'vcpill',
    'wooden_beads', 'woodstick', 
    ]

    def __init__(self, root='data/realAD'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test']

    def run(self):
        """
        扫描RealAD数据集目录结构并生成meta.json
        目录结构: root/category/category/OK|NG/...
        """
        info = {phase: {cls_name: [] for cls_name in self.CLSNAMES} for phase in self.phases}
        anomaly_samples = 0
        normal_samples = 0

        for cls_name in self.CLSNAMES:
            # RealAD的路径结构: root/category/category/OK|NG/...
            cls_dir = os.path.join(self.root, cls_name, cls_name)
            
            if not os.path.exists(cls_dir):
                print(f"警告: 类别目录不存在: {cls_dir}")
                continue

            # 遍历OK和NG目录
            for status in ['OK', 'NG']:
                status_dir = os.path.join(cls_dir, status)
                if not os.path.exists(status_dir):
                    continue

                # 遍历样本目录（如S0001, S0002等）
                for sample_dir in sorted(os.listdir(status_dir)):
                    sample_path = os.path.join(status_dir, sample_dir)
                    if not os.path.isdir(sample_path):
                        continue

                    # 遍历该样本的所有图像
                    for img_file in sorted(os.listdir(sample_path)):
                        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                            continue

                        # 构建相对路径
                        img_relative_path = os.path.join(cls_name, status, sample_dir, img_file)
                        
                        # 查找对应的mask
                        mask_relative_path = ''
                        if status == 'NG':
                            # mask文件名通常是将.jpg替换为.png
                            mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                            mask_path = os.path.join(sample_path, mask_file)
                            if os.path.exists(mask_path):
                                mask_relative_path = os.path.join(cls_name, status, sample_dir, mask_file)

                        # 判断是否为异常
                        is_abnormal = (status == 'NG' and mask_relative_path != '')
                        
                        info_img = dict(
                            img_path=img_relative_path,
                            mask_path=mask_relative_path,
                            cls_name=cls_name,
                            specie_name=status,
                            anomaly=1 if is_abnormal else 0,
                        )
                        
                        # RealAD只有测试集
                        info['test'][cls_name].append(info_img)
                        
                        if is_abnormal:
                            anomaly_samples += 1
                        else:
                            normal_samples += 1

        # 保存meta.json
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
        
        print(f'生成完成！')
        print(f'正常样本: {normal_samples}')
        print(f'异常样本: {anomaly_samples}')
        print(f'总样本数: {normal_samples + anomaly_samples}')
        print(f'meta.json保存在: {self.meta_path}')


if __name__ == '__main__':
    # 修改为你的RealAD数据集路径
    runner = RealADSolver(root='/path/to/realiad_512')
    runner.run()
