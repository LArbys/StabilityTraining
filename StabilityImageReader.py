import os,sys
from larcv import larcv
import numpy as np
import tensorflow as tf
import threading

# this is the function that adds noise to the noisy image. All sorts of things to mess with here. 
# my version here does 4 things
# swap around columns and rows with nearby rows and columns, add gaussian noise and adjust the global brightness of the image

#besides adjusting the noise module, there isnt much to mess with here

def image_manipulation(outimg):
    vec = outimg.shape
    top = 10 # for switching around columns and rows. how many do you wanna switch?
    choicea = np.random.randint(4) #do you wanna switch with every image? Currently set for 1 in 4
    if choicea ==0:
        for i in range(top):                                                                                                           
            choice = np.random.randint(vec[0])                                                                                             
            choice3 = np.random.randint(vec[1])                                                                                            
            choice2 = np.random.randint(5)                                                                                                 
            index = choice3-2+choice2                                                                                                      
            if index>=vec[1]-1:                                                                                                            
                index = vec[1]-1                                                                                                           
            outimg[choice][choice3] = outimg[choice][index]                                                                                
    choiceb = np.random.randint(4) # switching around rows
    if choiceb ==0:
        for j in range(top):
            choice = np.random.randint(vec[0])
            choice3 = np.random.randint(vec[1])
            choice2 = np.random.randint(5)
            index = choice3-2+choice2
            if index>=vec[0]-1:
                index = vec[0]-1
            outimg[choice][choice3] = outimg[index][choice3]
#    choicec =  np.random.randint(3) how often do you wanna add gaussian noise?
    choicec = 0 #currently I am always adding Gaussian noise. 
    if choicec ==0:
        var = np.random.uniform(0.01,0.5) # I've found varying variance to be useful. I trained on 0.01 to 0.5
        adding = var*np.random.randn(vec[0],vec[1],vec[2])
        outimg = outimg + adding

    choiced == 1 #currently brightness is turned off
    if choiced == 0:
        brighthi = np.random.uniform(70,120) # adjust brightness between 70% and 120 %
        bright = brighthi/100.0
        outimg = bright*outimg

    


#    outimg[outimg<0.2]=0      for visualizing images, you dont want any negative values (which happens if you add gaussian noise to numbers near zero because it wraps around from being bright to very dark. This function makes the images pretty. Basically it keeps the background black

    return outimg

class Image2DReader:
    """Class that provides image data to TensorFlow Models. Tailored to classificaion."""
    def __init__(self,drivername,cfg,batch_size,nclasses,filelist=[],loadflat=False):
        """
        constructor

        inputs:
        drivername: string wuth process driver name
        cfg: string containing path to ProcessDriver configuration file
        filelist: list of strings containing path to input files. config will often provide filelist.
        """
        # check variable types
        assert type(drivername) is str
        assert type(cfg) is str
        assert type(batch_size) is int
        self.loadflat = loadflat
        self.batch_size = batch_size
        self.drivername = drivername
        self.nclasses = nclasses
#        self.counting = 30000
        # setup process driver
        self.config_file = cfg
        filler_exists = larcv.ThreadFillerFactory.exist_filler(self.drivername)
        if not filler_exists:
            print "Get Filler: ",self.drivername
            self.proc = larcv.ThreadFillerFactory.get_filler(self.drivername)
            self.proc.configure(self.config_file)
        else:
            print "Filler Already Exists"

        self.get_image_attributes()
        self.defineSubNetwork()
        print "Image2DReader network defined. Get image and lable Tensor variables via:"
        print " images: get_image_batch_node()"
        print " labels: get_label_batch_node()"

    def setfilelist(self, filelist ):
        assert filelist is list
        # do i want to check if files exist?
        self.proc.override_input_file( filelist )
        # re-initialize
        self.proc.initialize()

    def get_image_attributes(self):
        self.proc.set_next_index(0)
        self.proc.batch_process( 1 )
        dims = self.proc.dim()
        self.nchs = dims.at(1)
        self.rows = dims.at(2)
        self.cols = dims.at(3)
        self.vecshape = self.nchs*self.rows*self.cols
        self.proc.set_next_index(0)
        
    def load_data_worker( self ):
        # at start we need image shape size, process first image
        # start from beginning
        self.proc.set_next_index(0)
        while True:
            self.proc.batch_process( 1 ) #self.batch_size )
            data = self.proc.data_ndarray() # 1D data (for all batch size)
            label = self.proc.labels()
            #outimg = np.zeros( (self.vecshape,), dtype=np.float32 )
            #outimg = data # copy
            #outimg = np.transpose( outimg.reshape( (self.nchs, self.rows, self.cols) ), (1,2,0) ) # change from CHW to HWC (more natural for TF)
            data = np.transpose( data.reshape( (self.nchs, self.rows, self.cols) ), (1,2,0) ) # change from CHW to HWC (more natural for TF)
            outimg = data # here is the "original image"
            manip = image_manipulation(outimg) # here is its cousin, "noisy image"
            outlabel = np.zeros( (self.nclasses,), dtype=np.float32 )
            outlabel[label.at(0)] = 1.0
            #print "Ask process driver for batch",label.at(0)

# now I shove the label, image and noisy image into the feed queue so it pops up in my network

            if self.loadflat:
                self.tfsession.run( self.enqueue_op, feed_dict={self.ph_enqueue_image:outimg.flatten(),self.ph_enqueue_label:outlabel, self.ph_enqueue_manip:manip.flatten()} )
            else:
                self.tfsession.run( self.enqueue_op, feed_dict={self.ph_enqueue_image:outimg,self.ph_enqueue_label:outlabel, self.ph_enqueue_manip:manip} )                

    def defineSubNetwork(self):

        # setup network
        with tf.name_scope('image2dreader_'+self.drivername):
            if self.loadflat:
                self.ph_enqueue_image = tf.placeholder(tf.float32, shape=[self.vecshape], name="Enqueue_Image_"+self.drivername)
                self.ph_enqueue_manip = tf.placeholder(tf.float32, shape=[self.vecshape], name="Enqueue_Image_"+self.drivername)
            else:
                self.ph_enqueue_image = tf.placeholder(tf.float32, shape=[self.rows,self.cols,self.nchs], name="Enqueue_Image_"+self.drivername)
                self.ph_enqueue_manip = tf.placeholder(tf.float32, shape=[self.rows,self.cols,self.nchs], name="Enqueue_Image_"+self.drivername)

            self.ph_enqueue_label = tf.placeholder(tf.float32, shape=[self.nclasses],name="Enqueue_Label_"+self.drivername)
            self.example_queue = tf.FIFOQueue( capacity=3*self.batch_size, dtypes=[tf.float32, tf.float32, tf.float32], shapes=[[self.rows,self.cols,self.nchs], [self.nclasses],[self.rows,self.cols,self.nchs]] )
            self.enqueue_op = self.example_queue.enqueue([self.ph_enqueue_image, self.ph_enqueue_label, self.ph_enqueue_manip])
            self.image_batch, self.label_batch, self.manip_batch = self.example_queue.dequeue_many(self.batch_size)

    def startQueue( self, tfsession, batch_size ):
        """ Starts the image2dreader sub-network. returns the placeholder variables to give to model
        inputs:
        tfsession: tensorflow session
        """
        # store pointers
        self.tfsession = tfsession
        #self.batch_size = batch_size
        #self.defineSubNetwork()

        self.worker_thread = threading.Thread( target=self.load_data_worker )
        self.worker_thread.daemon = True
        self.worker_thread.start()

        return self.image_batch, self.label_batch, self.manip_batch

    def get_image_batch_node(self):
        if not hasattr(self, 'image_batch'):
            raise RuntimeError("Must call startQueue first for image batch to be created")
        return self.image_batch

    def get_manip_batch_node(self):
        if not hasattr(self, 'manip_batch'):
            raise RuntimeError("Must call startQueue first for image batch to be created")
        return self.manip_batch


    def get_label_batch_node(self):
        if not hasattr(self, 'label_batch'):
            raise RuntimeError("Must call startQueue first for label_batch to be created")
        return self.label_batch

    def get_image_shape(self,order='HWC'):
        if order=='HWC':
            return [self.rows,self.cols,self.nchs]
        elif order=='CHW':
            return [self.nchs,self.rows,self.cols]
        else:
            raise ValueError('order must be \'HWC\' or \'CHW\'')
