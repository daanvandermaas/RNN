library(tensorflow)
library(rlist)

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 0
batch_size = 5
num_batches = floor(total_series_length/(batch_size * truncated_backprop_length))
lrate = 0.1


#maak dumydata
generateData = function(){
  x = sample(c(1,0), total_series_length, replace = TRUE )  
  y = c(1:echo_step, x)[1:total_series_length]
  y[1:echo_step] = 0
  
  x = matrix(x, nrow = batch_size)
  y = matrix(y, nrow = batch_size)
  
  return(list(x,y))
}


#de placeholders
batchX_placeholder = tf$placeholder(tf$float32, c(batch_size, truncated_backprop_length))
batchY_placeholder = tf$placeholder(tf$int32, c(batch_size, truncated_backprop_length))



#de variabelen
W = tf$Variable(tf$truncated_normal(shape(1L, num_classes), stddev=0.1))
b = tf$Variable(tf$truncated_normal(shape(1L, num_classes), stddev=0.1))

#unpack batchX en batchY
inputs_series = tf$unstack(batchX_placeholder, axis = 1L)
labels_series = tf$unstack(batchY_placeholder, axis = 1L)

#Forwardpass
logits_series = list()
for(input in inputs_series){
  input = tf$reshape(input, shape = shape(batch_size,1L))
  logit = tf$matmul(input,W) +b
  logits_series = c(logits_series, logit)
}




#softmax
prediction_series = list()
for(logit in logits_series){
  prediction = tf$nn$softmax(logit)
  prediction_series =  c(prediction, prediction_series)
}




cost = list()
for( i in 1:length(prediction_series)){
  cost = c(    cost,   tf$nn$sparse_softmax_cross_entropy_with_logits(   logits = prediction_series[[i]], labels = labels_series[[i]]) )
}

loss = tf$reduce_mean( tf$stack(cost, axis = 0))

train_step = tf$train$AdamOptimizer(lrate)$minimize(loss)


#start sessie
sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())




for(epoch_idx in 1:num_epochs){
  x = generateData()
  y = x[[2]]
  x = x[[1]]
  
  print( paste('new data, epoch', epoch_idx))
  
  for(batch_idx in 1:num_batches){
    start_idx = (batch_idx-1)*truncated_backprop_length +1
    end_idx = start_idx + truncated_backprop_length -1
    
    batchX = x[,start_idx:end_idx]
    batchY = y[,start_idx:end_idx]
    batchY = array( c(as.numeric(batchY == 0), as.numeric(batchY == 1)) , dim = c(dim(batchX), 2) )
    
    
    
    train_step$run(feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY))
    
    
    
  }
  
  prestatie = sess$run(loss ,feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY))
  print(paste('cost:',prestatie))
  
}




sess$run(  labels_series[[1]] , feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY, init_state_placeholder = init_state))

