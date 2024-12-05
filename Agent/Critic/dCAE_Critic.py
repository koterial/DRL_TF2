import tensorflow as tf
import tensorflow.keras as keras

tf.keras.backend.set_floatx('float32')


# 用于拟合状态语义价值H(s)并重构状态的Critic模型, 可更好的平衡探索与利用, 论文《Efficient Episodic Memory Utilization of Cooperative Multi-Agent Reinforcement Learning》
# 实例化参数：状态空间维度, 值空间维度, Encoder网络结构, Encoder网络结构, 输出端激活函数
class dCAE_Critic():
    def __init__(self, state_shape, value_shape, encode_unit_num_list, decode_unit_num_list, activation):
        self.state_shape = state_shape
        self.value_shape = value_shape
        self.encode_unit_num_list = encode_unit_num_list
        self.decode_unit_num_list = decode_unit_num_list
        self.activation = activation
        self.model = self.model_create()

    def model_create(self):
        # 创建状态输入端
        self.state_input_layer = [
            keras.Input(shape=sum(self.state_shape), name="critic_state_input")
        ]
        # 创建时序输入端
        self.time_input_layer = [
            keras.Input(shape=1, name="critic_time_input")
        ]
        # 创建输入链接层
        self.input_concat_layer = keras.layers.Concatenate(name="critic_input_concat")
        # 创建Encoder中间层
        self.encode_hidden_layer_list = [
            keras.layers.Dense(self.encode_unit_num_list[each], activation="relu", name="critic_encode_hidden_{}".format(each)) for each in range(len(self.encode_unit_num_list))
        ]
        # 创建Decoder中间层
        self.decode_hidden_layer_list = [
            keras.layers.Dense(self.decode_unit_num_list[0][each], activation="relu", name="critic_decode_hidden_{}".format(each)) for each in range(len(self.decode_unit_num_list[0]))
        ]
        # 创建状态输出端
        self.state_output_layer_list = [
            keras.layers.Dense(self.decode_unit_num_list[1], activation="relu", name="critic_decode_state_hidden"),
            keras.layers.Dense(sum(self.value_shape), activation=self.activation[0], name="critic_state_output")
        ]
        # 创建值输出端
        self.value_output_layer_list = [
            keras.layers.Dense(self.decode_unit_num_list[2], activation="relu", name="critic_decode_value_hidden"),
            keras.layers.Dense(sum(self.value_shape), activation=self.activation[1], name="critic_value_output")
        ]
        # 创建输出链接层
        self.output_concat_layer = keras.layers.Concatenate(name="critic_output_concat")
        x = self.input_concat_layer(self.state_input_layer + self.time_input_layer)
        for hidden_layer in self.encode_hidden_layer_list:
            x = hidden_layer(x)
        x = self.input_concat_layer([x] + self.time_input_layer)
        for hidden_layer in self.decode_hidden_layer_list:
            x = hidden_layer(x)
        output_list = []
        output_list.append(self.state_output_layer_list[1](self.state_output_layer_list[0](x)))
        output_list.append(self.value_output_layer_list[1](self.value_output_layer_list[0](x)))
        output = self.output_concat_layer(output_list)
        # 创建模型
        model = keras.Model(inputs=self.state_input_layer + self.time_input_layer, outputs=output)
        return model
        
    # 向前传播
    @tf.function
    def get_value(self, state_batch, time_batch):
        value_batch = self.model([state_batch, time_batch])
        return value_batch
    
if __name__ == "__main__":
    import numpy as np
    critic = dCAE_Critic(state_shape=[35], value_shape=[1], encode_unit_num_list=[32, 32, 32], decode_unit_num_list=[[32, 32, 32], 32, 32], activation=["tanh", "linear"])
    state_batch = np.random.uniform(size=(128, 35))
    time_batch = np.random.uniform(size=(128, 1))
    summary_writer = tf.summary.create_file_writer("Demo/dCAE_Critic/")
    tf.summary.trace_on(graph=True, profiler=True)
    with summary_writer.as_default():
        critic.get_value(state_batch, time_batch)
        tf.summary.trace_export(name="dCAE Critic Model", step=0, profiler_outdir="Demo/dCAE_Critic/")