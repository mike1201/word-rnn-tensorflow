# word-rnn-tensorflow

## Structure of Code

Utils.py - Class TextLoader()
# 1-1. Define Instance Variable : data_dir, batch_size, seq_length

# 1-2. Get whole route of data

# 1-3. Clean string

# 1-4. Build Vocab_Index Dictionary
- Builds a Vocabulary mapping from word to index using sentences
# 1-4-1. Input : sentences = [word1, word2, ...]
# 1-4-2. Make Word Set
# 1-4-3. Make Vocab_Dictionary

# 1-5. preprocess
# 1-5-1. read Input data
# 1-5-2. Clean Text(1-3) and split the string
# 1-5-3. Make Vocab_Index_Dictionary (1-4)
# 1-5-4. Save word set by ‘vocab_file’
# 1-5-5. Change string sentence to idx sentence ( array )
# 1-5-6. Save idx sentence by ‘data.npy’
# 1-6. load_preprocessed : Emit
# 1-7. Create Batches
# 1-7-1. Set number of batches per epoch.
- self.num_batches = int(self.tensor.size / self.seq_length * self.batch_size)
- tensor.size (Number of whole words) is divided by seq_length
- It means that RNN Cell SIze = Seq_length
# 1-7-2. Cut Residual Data
# 1-7-3. Define whole xdata, ydata
# 1-7-4. Make Batch Data
# 1-8 Return next batch
# 1-9 Reset batch_pointer to zero



# 2. model.py : class Model()
# 2-1. Define Instance Variable = args
# 2-2. Choose Model Type (rnn, gru, lstm)
# 2-3. Make multi cells
- Decide output size( =args.rnn_size =Embedding Size ) and Number of layers 
# 2-4. Set Input data
- Size = [batch_size, seq_length]
- Decide batch_size and seq_length
# 2-5. Define Tensor regarding ‘batch, epoch’ and Initial state( Co=[batch_size] )
** why Initial State`s size == batch_size? : It`s definitely true **
# 2-6. Define Softmax Weight and Bias Tensor
# 2-7. Define Embedding Matrix
- size = [vocab_size, args.rnn_size] (= [10000, 256] )
# 2-8. Change Input data
(1) self.input_data = [batch_size, args.seq_length] (= [50, 25] )
(2) Apply embedding to input_data 
   inputs = tf.nn.embedding_lookup(embedding, self.input_data) 
          = [50, 25, 256] = [batch, word, embedding variable]
(3) Split input data by word 
    inputs = tf.split(inputs, args.seq_length) = [[50,1,256], [50,1,256], ... 25th]
(4) tf.squeeze(inputs) = [[50,256], [50,256], [50,256], ..., 25th] = [50_Batch, 25_word] 
		     = [batch_size, seq_length, args.rnn_size]
** We have to change input data like this to put in legacy_seq2seq.rnn_decoder **
# 2-9. Get outputs and Cell state
(1) outputs = [batch_size, seq_length, args.rnn_size] ( = [batch, word, embeddings] )
--) output = tf.concat(outputs, 1) = [batch1, batch2, ... , batch50]
    ## Can`t understand
--) output = tf.reshape(output, [-1, args.rnn_size] ) = [batch_word, 256]
(2) last_state = [batch_size]
# 2-10. Define hypothesis and probability
# 2-11. Define Loss Function and final_state
- legacy_seq2seq.sequence_loss_by_example에 대해서 공부할 필요
# 2-12. Train Layer
- Define optimizer and Learning rate to apply decay rate.


Train,py
# 3-1. Define data_loader class = TextLoader
# 3-2. Make Model
# 3-3. Initialize the variable
# 3-4. Epoch Loop
- Set Decay Rate considering Epoch
- batch = 0
- zero-filled Cell state
# 3-5. Batch Loop
- Generate Batch Data
- Feed : batch_x, batch_y, state, speed
- Train : cost, final_state, train_op, inc_batch_pointer_op

Save Model
# 5-1. Define a filewriter
# 5-2. After making graph, add graph
# 5-3. Save Initial Values.
# 5-4. Save training process
# 5-5. Save
