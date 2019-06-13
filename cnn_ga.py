# 主函数
#导入要用到的模块
import paddle
import paddle.fluid as fluid
import numpy
import sys
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

import random
import operator

import os
import json

# ①testrate取0.05,0.1,0.2
def setDataList(test_rate):
    data_root_path = '/home/aistudio/data/data2394/images/face'
    class_detail = []
    class_dirs = os.listdir(data_root_path)
    class_label = 0
    father_paths = data_root_path.split('/')    
    while True:
        if father_paths[father_paths.__len__() - 1] == '':
            del father_paths[father_paths.__len__() - 1]
        else:
            break
    father_path = father_paths[father_paths.__len__() - 1]
    data_list_path = '/home/aistudio/data/data2394/%s/' % father_path
    isexist = os.path.exists(data_list_path)
    if not isexist:
        os.makedirs(data_list_path)
    with open(data_list_path + "test.list", 'w') as f:
        # f.truncate()
        pass
    with open(data_list_path + "trainer.list", 'w') as f:
        # f.truncate()
        pass
    # 总的图像数量
    all_class_images = 0
    # 读取每个类别
    for class_dir in class_dirs:#["jiangwen","pengyuyan","zhangziyi"]
        # 每个类别的信息
        class_detail_list = {}
        test_sum = 0
        trainer_sum = 0
        # 统计每个类别有多少张图片
        class_sum = 0
        # 获取类别路径
        path = data_root_path + "/" + class_dir
        # 获取所有图片
        img_paths = os.listdir(path)
    
        for img_path in img_paths:                                  # 遍历文件夹下的每个图片
            name_path = path + '/' + img_path                       # 每张图片的路径
            if class_sum % (1.0/test_rate) == 0:                                 # 每10张图片取一个做测试数据
                test_sum += 1                                       #test_sum测试数据的数目
                with open(data_list_path + "test.list", 'a') as f:#追加内容到test.list
                    f.write(name_path + "\t%d" % class_label + "\n") #class_label 标签：0,1,2
            else:
                trainer_sum += 1                                    #trainer_sum测试数据的数目
                with open(data_list_path + "trainer.list", 'a') as f:
                    f.write(name_path + "\t%d" % class_label + "\n")#class_label 标签：0,1,2
            class_sum += 1                                          #每类图片的数目
            all_class_images += 1                                   #所有类图片的数目
    
        # 说明的json文件的class_detail数据
        class_detail_list['class_name'] = class_dir             #类别名称，如jiangwen
        class_detail_list['class_label'] = class_label          #类别标签，0,1,2
        class_detail_list['class_test_images'] = test_sum       #该类数据的测试集数目
        class_detail_list['class_trainer_images'] = trainer_sum #该类数据的训练集数目
        class_detail.append(class_detail_list)         
        class_label += 1                                            #class_label 标签：0,1,2
    # 获取类别数量
    all_class_sum = class_dirs.__len__()
    # 说明的json文件信息
    readjson = {}
    readjson['all_class_name'] = father_path                  #文件父目录
    readjson['all_class_sum'] = all_class_sum                #
    readjson['all_class_images'] = all_class_images
    readjson['class_detail'] = class_detail
    jsons = json.dumps(readjson, sort_keys=True, indent=4, separators=(',', ': '))
    with open(data_list_path + "readme.json",'w') as f:
        f.write(jsons)
    print ('生成数据列表完成！')

# 定义训练的mapper
# train_mapper函数的作用是用来对训练集的图像进行处理修剪和数组变换，返回img数组和标签 
# sample是一个python元组，里面保存着图片的地址和标签。 ('../images/face/zhangziyi/20181206145348.png', 2)
def train_mapper(sample):
    img, label = sample
    # 进行图片的读取，由于数据集的像素维度各不相同，需要进一步处理对图像进行变换
    img = paddle.dataset.image.load_image(img)       
    #进行了简单的图像变换，这里对图像进行crop修剪操作，输出img的维度为(3, 100, 100)
    img = paddle.dataset.image.simple_transform(im=img,          #输入图片是HWC   
                                                resize_size=100, # 剪裁图片
                                                crop_size=100, 
                                                is_color=True,  #彩色图像
                                                is_train=True)
    #将img数组进行进行归一化处理，得到0到1之间的数值
    img= img.flatten().astype('float32')/255.0
    return img, label
# 对自定义数据集创建训练集train的reader
def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, 'r') as f:
            # 将train.list里面的标签和图片的地址方法一个list列表里面，中间用\t隔开'
            #../images/face/jiangwen/0b1937e2-f929-11e8-8a8a-005056c00008.jpg\t0'
            lines = [line.strip() for line in f]
            for line in lines:
                # 图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab) 
    # 创建自定义数据训练集的train_reader
    return paddle.reader.xmap_readers(train_mapper, reader,cpu_count(), buffered_size)

# sample是一个python元组，里面保存着图片的地址和标签。 ('../images/face/zhangziyi/20181206145348.png', 2)
def test_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(img)
    img = paddle.dataset.image.simple_transform(im=img, resize_size=100, crop_size=100, is_color=True, is_train=False)
    img= img.flatten().astype('float32')/255.0
    return img, label

# 对自定义数据集创建验证集test的reader
def test_r(test_list, buffered_size=1024):
    def reader():
        with open(test_list, 'r') as f:
            lines = [line.strip() for line in f]
            for line in lines:
                #图像的路径和标签是以\t来分割的,所以我们在生成这个列表的时候,使用\t就可以了
                img_path, lab = line.strip().split('\t')
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(test_mapper, reader,cpu_count(), buffered_size)
    

# 打印测试
# temp_reader = paddle.batch(trainer_reader,
#                             batch_size=3)
# temp_data=next(temp_reader())
# print(temp_data)

def convolutional_neural_network(image, type_size):
    # 第一个卷积--池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,# 输入图像
                                                       filter_size=3,# 滤波器的大小
                                                       num_filters=32,# filter 的数量。它与输出的通道相同
                                                       pool_size=2,# 池化层大小2*2
                                                       pool_stride=2,# 池化层步长
                                                       act='relu') # 激活类型
    
    # Dropout主要作用是减少过拟合，随机让某些权重不更新  
    # Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。
    # 根据给定的丢弃概率dropout随机将一些神经元输出设置为0，其他的仍保持不变。
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=0.5)
    
    # 第二个卷积--池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,
                                                       filter_size=3,
                                                       num_filters=64,
                                                       pool_size=2,
                                                       pool_stride=2,
                                                       act='relu')
    # 减少过拟合，随机让某些权重不更新                                                   
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=0.5)
    
    # 第三个卷积--池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,
                                                       filter_size=3,
                                                       num_filters=64,
                                                       pool_size=2,
                                                       pool_stride=2,
                                                       act='relu')
    # 减少过拟合，随机让某些权重不更新                                                   
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=0.5)
    
    # 全连接层
    fc = fluid.layers.fc(input=drop, size=512, act='relu')
    # 减少过拟合，随机让某些权重不更新                                                   
    drop =  fluid.layers.dropout(x=fc, dropout_prob=0.5)                                                   
    # 输出层 以softmax为激活函数的全连接输出层，输出层的大小为图像类别type_size个数
    predict = fluid.layers.fc(input=drop,size=type_size,act='softmax')
    
    return predict

def convolutional_neural_network_withpara(image, type_size,paralist):
    # 第一个卷积--池化层
    conv_pool_1 = fluid.nets.simple_img_conv_pool(input=image,# 输入图像
                                                       filter_size=paralist[0],# 滤波器的大小
                                                       num_filters=paralist[1],# filter 的数量。它与输出的通道相同
                                                       pool_size=paralist[2],# 池化层大小2*2
                                                       pool_stride=paralist[2],# 池化层步长
                                                       act=paralist[3]
                                                       ) # 激活类型
    
    # Dropout主要作用是减少过拟合，随机让某些权重不更新  
    # Dropout是一种正则化技术，通过在训练过程中阻止神经元节点间的联合适应性来减少过拟合。
    # 根据给定的丢弃概率dropout随机将一些神经元输出设置为0，其他的仍保持不变。
    drop = fluid.layers.dropout(x=conv_pool_1, dropout_prob=paralist[4])
    
    # 第二个卷积--池化层
    conv_pool_2 = fluid.nets.simple_img_conv_pool(input=drop,
                                                       filter_size=paralist[5],
                                                       num_filters=paralist[6],
                                                       pool_size=paralist[7],
                                                       pool_stride=paralist[7],
                                                       act=paralist[8])
    # 减少过拟合，随机让某些权重不更新                                                   
    drop = fluid.layers.dropout(x=conv_pool_2, dropout_prob=paralist[9])
    
    # 第三个卷积--池化层
    conv_pool_3 = fluid.nets.simple_img_conv_pool(input=drop,
                                                       filter_size=paralist[10],
                                                       num_filters=paralist[11],
                                                       pool_size=paralist[12],
                                                       pool_stride=paralist[12],
                                                       act=paralist[13])
    # 减少过拟合，随机让某些权重不更新                                                   
    drop = fluid.layers.dropout(x=conv_pool_3, dropout_prob=paralist[14])
    
    # 全连接层
    fc = fluid.layers.fc(input=drop, size=paralist[15], act=paralist[16])
    # 减少过拟合，随机让某些权重不更新                                                   
    drop =  fluid.layers.dropout(x=fc, dropout_prob=paralist[17])                                                   
    # 输出层 以softmax为激活函数的全连接输出层，输出层的大小为图像类别type_size个数
    predict = fluid.layers.fc(input=drop,size=type_size,act='softmax')
    
    return predict


def draw_train_process(title,iters,costs,accs,label_cost,lable_acc,path):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel("cost/acc", fontsize=20)
    plt.plot(iters, costs,color='red',label=label_cost) 
    plt.plot(iters, accs,color='green',label=lable_acc) 
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(path)

def startup_train(paralist):
    BATCH_SIZE = paralist[18]#32
    # 把图片数据生成reader
    trainer_reader = train_r(train_list="/home/aistudio/data/data2394/face/trainer.list")
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            reader=trainer_reader,buf_size=300),
        batch_size=BATCH_SIZE)
    
    tester_reader = test_r(test_list="/home/aistudio/data/data2394/face/test.list")
    test_reader = paddle.batch(
         tester_reader, batch_size=BATCH_SIZE)
    
    image = fluid.layers.data(name='image', shape=[3, 100, 100], dtype='float32')#[3, 100, 100]，表示为三通道，100*100的RGB图
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    print('image_shape:',image.shape)
    
    #获取分类器     
    # predict = convolutional_neural_network_withpara(image=image, type_size=3, paralist=paralist)
    predict = convolutional_neural_network(image=image, type_size=3)

    # 获取损失函数和准确率
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    # 计算cost中所有元素的平均值
    avg_cost = fluid.layers.mean(cost)
    #计算准确率
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    # 定义优化方法
    optimizer = fluid.optimizer.Adam(learning_rate=paralist[19])    # Adam是一阶基于梯度下降的算法，基于自适应低阶矩估计该函数实现了自适应矩估计优化器
    optimizer.minimize(avg_cost)                             # 取局部最优化的平均损失
    print(type(accuracy))
    
    # 使用CPU进行训练
    # place = fluid.CPUPlace()
    
    # 使用GPU进行训练,参数0指的是显卡序号
    place = fluid.CUDAPlace(0)
    # 创建一个executor
    exe = fluid.Executor(place)
    # 对program进行参数初始化1.网络模型2.损失函数3.优化函数
    exe.run(fluid.default_startup_program())
    # 定义输入数据的维度,DataFeeder 负责将reader(读取器)返回的数据转成一种特殊的数据结构，使它们可以输入到 Executor
    feeder = fluid.DataFeeder(feed_list=[image, label], place=place)#定义输入数据的维度，第一个是图片数据，第二个是图片对应的标签。
    
    all_train_iter=0
    all_train_iters=[]
    all_train_costs=[]
    all_train_accs=[]
    
    test_iter=0
    test_iters=[]
    test_accs = []                                                            #测试的损失值
    test_costs = []                                                           #测试的准确率
    
    # 训练的轮数final_train_acc
    EPOCH_NUM = paralist[20]
    print('开始训练...')
    final_train_acc=0
    final_test_acc=0
    for pass_id in range(EPOCH_NUM):
        train_cost = 0
        for batch_id, data in enumerate(train_reader()):                         #遍历train_reader的迭代器，并为数据加上索引batch_id
            train_cost, train_acc = exe.run(
                program=fluid.default_main_program(),                            #运行主程序
                feed=feeder.feed(data),                                          #喂入一个batch的数据
                fetch_list=[avg_cost, accuracy])                                 #fetch均方误差和准确率
            
            all_train_iter=all_train_iter+BATCH_SIZE
            all_train_iters.append(all_train_iter)
            all_train_costs.append(train_cost[0])
            all_train_accs.append(train_acc[0])
            
            final_train_acc=train_acc[0]
            
            if batch_id % 10 == 0:                                               #每10次batch打印一次训练、进行一次测试
                print("\nPass %d, Step %d, Cost %f, Acc %f" % 
                (pass_id, batch_id, train_cost[0], train_acc[0]))

        # 每训练一轮 进行一次测试
        for batch_id, data in enumerate(test_reader()):                           # 遍历test_reader
            test_cost, test_acc = exe.run(program=fluid.default_main_program(),  # #运行测试主程序
                                           feed=feeder.feed(data),                #喂入一个batch的数据
                                           fetch_list=[avg_cost, accuracy])       #fetch均方误差、准确率
            test_iter=test_iter+BATCH_SIZE
            test_iters.append(test_iter)
            test_accs.append(test_acc[0])                                        #记录每个batch的误差
            test_costs.append(test_cost[0])                                      #记录每个batch的准确率
            
            final_test_acc=test_acc[0]
    
        test_cost = (sum(test_costs) / len(test_costs))                           # 每轮的平均误差
        test_acc = (sum(test_accs) / len(test_accs))                              # 每轮的平均准确率
        print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))
        
        #两种方法，用两个不同的路径分别保存训练的模型
        # model_save_dir = "/home/aistudio/data/data2394/model_vgg"
        model_save_dir = "/home/aistudio/data/data2394/model_cnn"
        # 如果保存路径不存在就创建
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        # 保存训练的模型，executor 把所有相关参数保存到 dirname 中
        fluid.io.save_inference_model(dirname=model_save_dir, 
                                        feeded_var_names=["image"],
                                        target_vars=[predict],
                                        executor=exe)
    draw_train_process("training",all_train_iters,all_train_costs,all_train_accs,"trainning cost","trainning acc","/home/aistudio/pic/trainning_%s.png"%str(paralist))
    draw_train_process("testing",test_iters,test_costs,test_accs,"testing cost","testing acc","/home/aistudio/pic/testing_%s.png"%str(paralist))

    print('训练模型保存完成！')
    result=[]
    result.append(final_train_acc)
    result.append(final_test_acc)
    print(str(result)+'\n')
    return result
    
def start(para_lists):
    print(str(para_lists))
    # para_lists=[3,32,2,'relu',0.5,3,64,2,'relu',0.5,3,64,2,'relu',0.5,512,'relu',0.5,32,0.001,20,0.1]
    setDataList(para_lists[-1])
    return startup_train(para_lists)
    # setDataList(test_rate=0.1)
    # startup_train(BATCH_SIZE=32)
# start()

class Individual:
    
    c_kernel=[3,5,7]
    p_size_stride=[1,2]
    act=['relu']
    dropout=[0.4,0.5,0.6]
    c_filter_num=[8,16,32,64,128]
    fc_size=[512,256]
    batch_size=[16,32,48,64]
    # learning_rate=0.001+random.randint(0,99)*0.001#0.001-0.1
    learning_rate=[0.001,0.01,0.1]
    # epoch_num=5+random.randint(0,15)#5-20轮
    epoch_num=[10]
    test_rate=[0.05,0.1,0.2]#数据集中测试数据占的比例
    
    chromosome=[]
    fitness=-1
    def __init__(self):
        self.fitness=-1
        self.chromosome=[
            self.c_kernel[random.randint(0,len(self.c_kernel)-1)],
            self.c_filter_num[random.randint(0,len(self.c_filter_num)-1)],
            self.p_size_stride[random.randint(0,len(self.p_size_stride)-1)],
            self.act[random.randint(0,len(self.act)-1)],
            self.dropout[random.randint(0,len(self.dropout)-1)],
            self.c_kernel[random.randint(0,len(self.c_kernel)-1)],
            self.c_filter_num[random.randint(0,len(self.c_filter_num)-1)],
            self.p_size_stride[random.randint(0,len(self.p_size_stride)-1)],
            self.act[random.randint(0,len(self.act)-1)],
            self.dropout[random.randint(0,len(self.dropout)-1)],
            self.c_kernel[random.randint(0,len(self.c_kernel)-1)],
            self.c_filter_num[random.randint(0,len(self.c_filter_num)-1)],
            self.p_size_stride[random.randint(0,len(self.p_size_stride)-1)],
            self.act[random.randint(0,len(self.act)-1)],
            self.dropout[random.randint(0,len(self.dropout)-1)],
            self.fc_size[random.randint(0,len(self.fc_size)-1)],
            self.act[random.randint(0,len(self.act)-1)],
            self.dropout[random.randint(0,len(self.dropout)-1)],
            self.batch_size[random.randint(0,len(self.batch_size)-1)],
            self.learning_rate[random.randint(0,len(self.learning_rate)-1)],
            self.epoch_num[random.randint(0,len(self.epoch_num)-1)],
            self.test_rate[random.randint(0,len(self.test_rate)-1)]
            ]
        # self.chromosome=[3,32,2,'relu',0.5,3,64,2,'relu',0.5,3,64,2,'relu',0.5,512,'relu',0.5,32,0.001,20,0.1]
    def setFitness(self,fitness):
        self.fitness=fitness
    def getFitness(self):
        return self.fitness
    def setGene(self,offset,gene):
        self.chromosome[offset]=gene
    def getGene(self,offset):
        return self.chromosome[offset]
    def getChromosome(self):
        return list(self.chromosome)

class Population:
    individuals=[]
    populationFitness=-1
    def __init__(self,population_size):
        for i in range(0,population_size):
            self.individuals.append(Individual())
    def getIndivudials(self):
        return self.individuals
    
    def getIndivudial(self,offset):
        return self.individuals[offset]
    
    def setIndivudials(self,offset,individual):
        self.individuals[offset]=individual
        
    def getPopulationFitness(self):
        return self.populationFitness
    def setPopulationFitness(self,populationFitness):
        self.populationFitness=populationFitness
    
    def getFittest(self,offset):#按照偏移量得到较好的个体
        cmpfun = operator.attrgetter('fitness')#参数为排序依据的属性，可以有多个，这里优先id，使用时按需求改换参数即可
        self.individuals.sort(key=cmpfun)#排序,从小打大
        return self.individuals[len(self.individuals)-1-offset]
class GA:
    population_size=100
    mutation_rate=0.001
    crossover_rate=0.9
    elitismCount=0
    def __init__(self,population_size,mutation_rate,crossover_rate,elitismCount):
        self.population_size=population_size
        self.mutation_rate=mutation_rate
        self.crossover_rate=crossover_rate
        self.elitismCount=elitismCount
    def init_population(self):
        population=Population(self.population_size)
        return population
    def eval_population(self,population):
        for individual in population.getIndivudials():
            result=start(individual.chromosome)
            fitness=0.5*result[0]+result[1]*0.5
            individual.setFitness(fitness)
            
    def select_by_wheel(population):
        # todo：population中的individual应该根据适应度排序
        fitness=[]
        for individual in population:
            fitness.append(individual.getFitness())
        sumFits = sum(fitness)
        # 产生一个随机实数
        rndPoint = random.uniform(0, sumFits)
        # 计算得到应选择个体的index，返回对应个体
        accumulator = 0.0
        for ind, val in enumerate(fitness):
            accumulator += val
            if accumulator >= rndPoint:
                return population.getIndivudial[ind]

    def crossover_population(self,population):
        new_population=Population(len(population.getIndivudials()))
        for index,individual in enumerate(population):
            parent1=population.getFittest(index)
            if(self.crossover_rate>random.random() and index>= self.elitismCount):
                parent2=select_by_wheel(population)
                offspring=Individual()
                for index2,gene in enumerate(parent1):
                    if 0.5>random.random():
                        offspring.setGene(index2,parent1.getGene(index2))
                    elif 0.5<=random.random():
                        offspring.setGene(index2,parent2.getGene(index2))
                new_population.setIndivudial(index,offspring)
            new_population.setIndivudial(index,parent1)
        return new_population
    def mutate_population(self,population):
        c_kernel=[3,5,7]
        p_size_stride=[1,2]
        act=['relu']
        dropout=[0.4,0.5,0.6]
        c_filter_num=[8,16,32,64,128]
        fc_size=[512,256]
        batch_size=[16,32,48,64]
        # learning_rate=0.001+random.randint(0,99)*0.001#0.001-0.1
        learning_rate=[0.001,0.01,0.1]
        # epoch_num=5+random.randint(0,15)#5-20轮
        epoch_num=[10]
        test_rate=[0.05,0.1,0.2]#数据集中测试数据占的比例
        # [7, 128, 1, 'relu', 0.4, 5, 32, 1, 'relu', 0.6, 3, 8, 2, 'relu', 0.4, 256, 'relu', 0.4, 48, 0.01, 10, 0.1]
        new_population=Population(len(population.getIndivudials()))
        for index,individual in enumerate(population):
            temp_individual=population.getFittest(index)
            for index2,gene in enumerate(individual):
                if(self.mutation_rate>random.random() and index> self.elitismCount):
                    # todo 变异
                    if index<15:
                        if index%5==0:
                            temp_individual.setGene(index,c_kernel[random.randint(0,len(c_kernel)-1)])
                        elif index%5==1:
                            temp_individual.setGene(index,c_filter_num[random.randint(0,len(c_filter_num)-1)])
                        elif index%5==2:
                            temp_individual.setGene(index,p_size_stride[random.randint(0,len(p_size_stride)-1)])
                        elif index%5==3:
                            temp_individual.setGene(index,act[random.randint(0,len(act)-1)])
                        elif index%5==4:
                            temp_individual.setGene(index,dropout[random.randint(0,len(dropout)-1)])
                        else:
                            pass
                    elif  index==15:
                        temp_individual.setGene(index,fc_size[random.randint(0,len(fc_size)-1)])
                    elif  index==16:
                        temp_individual.setGene(index,act[random.randint(0,len(act)-1)])
                    elif  index==17:
                        temp_individual.setGene(index,dropout[random.randint(0,len(dropout)-1)])
                    elif  index==18:
                        temp_individual.setGene(index,batch_size[random.randint(0,len(batch_size)-1)])
                    elif  index==19:
                        temp_individual.setGene(index,learning_rate[random.randint(0,len(learning_rate)-1)])
                    elif  index==20:
                        temp_individual.setGene(index,epoch_num[random.randint(0,len(epoch_num)-1)])
                    else:
                        pass
            new_population.setIndivudial(index,temp_individual)    
        return new_population
    def met_termination_condition(self,population,opt_individual):
        individual=population.getFittest(0)
        if (opt_individual.getFitness()-individual.getFitness) <0.01:
            return 1
        else :
            return 0
        
def Drive():
    ga=GA(50,0.01,0.9,10)
    population=ga.init_population()
    ga.eval_population(population)
    generation=1
    opt_individual=Individual()
    count=1
    while ga.met_termination_condition(population,opt_individual)==0 or count<10:
        if  (opt_individual.getFitness()-population.getFittest(0).getFitness) >0:
            opt_individual=population.getFitness(0)
            count=1
        else:
            count+=1    
        population=ga.crossover_population(population);
        population=ga.mutate_population(population)
        ga.eval_population()
        generation+=1
        
    chromosome=population.getFittest(0).getChromosome
    print(str(generation))
    print(str(chromosome))
Drive()
    # indiv=Individual()
    # print(str(indiv.chromosome))
    # ff=start(indiv.chromosome)
    # print(str(ff))
