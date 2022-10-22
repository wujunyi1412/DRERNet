import torch as t
from Dataprocess.RGBT_dataprocessing_rail import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
from datetime import datetime

test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader2 = DataLoader(testData2, batch_size=1, shuffle=False, num_workers=4)


from Net.model.VAN.van_backbone_jia1_resnet import net2


net = net2()
net.load_state_dict(t.load('../../../Pth/van_backbone_jia1_resnet_rail_2022_06_07_10_58_best.pth'))   ########gaiyixia

a = '/home/wjy/文档/RGBD_rail/SalMap/'
b = 'van_backbone_jia1_resnet_rail_2022_06_07_10_58_best' ##########gaiyixia
c = '/rail_362/'
d = '/nlpr/'
e = '/nju2k/'

aa = []

vt800 = a + b + c
#
# nlpr = a + b + d
# nju2k = a + b + e



path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader1):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()


		out1, out2, out3, out4= net(image, depth)
		out = torch.sigmoid(out1)
		# out1 = torch.sigmoid(out2)
		# out2 = torch.sigmoid(out3)
		# out3 = torch.sigmoid(out4)

#
#
		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()
		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		print(path1 + name + '.png')
#
# 		# out_img = out1.cpu().detach().numpy()
# 		# out_img = out_img.squeeze()
# 		# plt.imsave(path1 + name + 'out1.png', arr=out_img, cmap='gray')
# 		#
# 		# out_img = out2.cpu().detach().numpy()
# 		# out_img = out_img.squeeze()
# 		# plt.imsave(path1 + name + 'out2.png', arr=out_img, cmap='gray')
# 		#
# 		# out_img = out3.cpu().detach().numpy()
# 		# out_img = out_img.squeeze()
# 		# plt.imsave(path1 + name + 'out3.png', arr=out_img, cmap='gray')
# 		#
# 		# out_img = out4.cpu().detach().numpy()
# 		# out_img = out_img.squeeze()
# 		# plt.imsave(path1 + name + 'out4.png', arr=out_img, cmap='gray')




##########################################################################################





