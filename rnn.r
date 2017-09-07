library(tensorflow)
library(rlist)

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 1
batch_size = 5
num_batches = floor(total_series_length/(batch_size * truncated_backprop_length))
lrate = 0.1


#maak dumydata
generateData = function(){
x = sample(c(1,0), total_series_length, replace = TRUE )  
x = matrix(x, nrow = batch_size, byrow = TRUE)

y = array(0, dim = c(nrow(x),echo_step))
y = cbind(y ,x)
y= y[,-c(ncol(y):(ncol(y)-echo_step)  )]


return(list(x,y))
}


#de placeholders
batchX_placeholder = tf$placeholder(tf$float32, c(batch_size, truncated_backprop_length))
batchY_placeholder = tf$placeholder(tf$int32, c(batch_size, truncated_backprop_length))

init_state_placeholder = tf$placeholder(tf$float32, c(batch_size, state_size))


#de variabelen
W_cell = tf$Variable(tf$truncated_normal(shape(state_size+1 ,state_size), stddev=0.1))
b_cell = tf$Variable(tf$truncated_normal(shape(1L, state_size), stddev=0.1))

W = tf$Variable(tf$truncated_normal(shape(state_size, num_classes), stddev=0.1))
b = tf$Variable(tf$truncated_normal(shape(1L, num_classes), stddev=0.1))

#unpack batchX en batchY
inputs_series = tf$unstack(batchX_placeholder, axis = 1L)
labels_series = tf$unstack(batchY_placeholder, axis = 1L)

#Forwardpass

#recurent layer
current_state = init_state_placeholder
states_series = list()
for(current_input in inputs_series){
current_input = tf$reshape(current_input, shape =  shape(batch_size, 1))
input_and_state_concatenated =  tf$concat(axis = 1L, values = list(current_input, current_state))
 
next_state = tf$tanh( tf$add(   tf$matmul(input_and_state_concatenated, W_cell  ), b_cell )  )
states_series = c(states_series, next_state)
current_state = next_state
 
}

#final layer
logits_series = list()
for(state in states_series){
logit = tf$matmul(state,W) +b
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

cost = c(cost, tf$nn$sparse_softmax_cross_entropy_with_logits(logits = logits_series[[i]], labels = labels_series[[i]]) )

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
  
  init_state = matrix(0, nrow = batch_size , ncol = state_size )
  
  print( paste('new data, epoch', epoch_idx))
  
  for(batch_idx in 1:num_batches){
    start_idx = (batch_idx-1)*truncated_backprop_length +1
    end_idx = start_idx + truncated_backprop_length -1
    
 batchX = x[,start_idx:end_idx]
 batchY = y[,start_idx:end_idx]

 
 
 train_step$run(feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY, init_state_placeholder = init_state))
 
 
    
  }
  
  prestatie = sess$run(loss ,feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY, init_state_placeholder = init_state))
  print(paste('cost:',prestatie))
  
}
  
  





sess$run(  loss , feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY, init_state_placeholder = init_state))



