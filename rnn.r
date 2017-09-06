library(tensorflow)
library(rlist)

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length/(batch_size * truncated_backprop_length)
lrate = 0.3


#maak dumydata
generateData = function(){
x = sample(c(1,0), total_series_length, replace = TRUE )  
y = c(1:echo_step, x)[1:total_series_length]
y[1:echo_step] = 0

return(list(x,y))
}


#de placeholders
batchX_placeholder = tf$placeholder(tf$float32, c(batch_size, truncated_backprop_length))
batchY_placeholder = tf$placeholder(tf$float32, c(batch_size, truncated_backprop_length, num_classes))

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
logit = tf$matmul(states_series[[1]],W) +b
logits_series = c(logits_series, logit)
}

#softmax
prediction_series = list()
for(logit in logits_series){
prediction = tf$nn$softmax(logit)
prediction_series =  c(prediction, prediction_series)
}




cost = 0
for( i in 1:length(logits_series)){
cost =   tf$nn$softmax_cross_entropy_with_logits( logits = logits_series[[1]], labels = labels_series[[1]]  )
}

train_step = tf$train$AdamOptimizer(lrate)$minimize(cost)


#start sessie
sess <- tf$InteractiveSession()
sess$run(tf$global_variables_initializer())

x = generateData()
y = x[[2]]
x = x[[1]]

batchX = x[1: (batch_size*truncated_backprop_length)]
batchY = y[1: (batch_size*truncated_backprop_length)]
batchX = matrix(batchX, nrow = batch_size)
batchY = matrix(batchY, nrow = batch_size)
batchY = array(c(as.numeric(batchY==0), as.numeric(batchY==1)), dim = c(dim(batchY), 2) )


init_state = matrix(0, nrow = batch_size , ncol = state_size )

for(i in 1:10000){
train_step$run(feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY, init_state_placeholder = init_state))
 c = sess$run(  cost , feed_dict = dict(batchX_placeholder = batchX, batchY_placeholder = batchY, init_state_placeholder = init_state))
print(c)
 }




