¿
ý
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.1.02unknown8¾À

my_model/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*(
shared_namemy_model/dense_4/kernel

+my_model/dense_4/kernel/Read/ReadVariableOpReadVariableOpmy_model/dense_4/kernel*
_output_shapes
:	@*
dtype0

my_model/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namemy_model/dense_4/bias
{
)my_model/dense_4/bias/Read/ReadVariableOpReadVariableOpmy_model/dense_4/bias*
_output_shapes
:@*
dtype0

my_model/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*(
shared_namemy_model/dense_5/kernel

+my_model/dense_5/kernel/Read/ReadVariableOpReadVariableOpmy_model/dense_5/kernel*
_output_shapes

:@
*
dtype0

my_model/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_namemy_model/dense_5/bias
{
)my_model/dense_5/bias/Read/ReadVariableOpReadVariableOpmy_model/dense_5/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/my_model/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*/
shared_name Adam/my_model/dense_4/kernel/m

2Adam/my_model/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_4/kernel/m*
_output_shapes
:	@*
dtype0

Adam/my_model/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/my_model/dense_4/bias/m

0Adam/my_model/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_4/bias/m*
_output_shapes
:@*
dtype0

Adam/my_model/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*/
shared_name Adam/my_model/dense_5/kernel/m

2Adam/my_model/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_5/kernel/m*
_output_shapes

:@
*
dtype0

Adam/my_model/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/my_model/dense_5/bias/m

0Adam/my_model/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_5/bias/m*
_output_shapes
:
*
dtype0

Adam/my_model/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*/
shared_name Adam/my_model/dense_4/kernel/v

2Adam/my_model/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_4/kernel/v*
_output_shapes
:	@*
dtype0

Adam/my_model/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/my_model/dense_4/bias/v

0Adam/my_model/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_4/bias/v*
_output_shapes
:@*
dtype0

Adam/my_model/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*/
shared_name Adam/my_model/dense_5/kernel/v

2Adam/my_model/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_5/kernel/v*
_output_shapes

:@
*
dtype0

Adam/my_model/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*-
shared_nameAdam/my_model/dense_5/bias/v

0Adam/my_model/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/my_model/dense_5/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
Ó
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bú


dense1

dense2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api

iter

beta_1

beta_2
	decay
learning_rate	m2
m3m4m5	v6
v7v8v9
 

	0

1
2
3

	0

1
2
3

non_trainable_variables
regularization_losses
layer_regularization_losses

layers
	variables
trainable_variables
metrics
 
US
VARIABLE_VALUEmy_model/dense_4/kernel(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model/dense_4/bias&dense1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1

non_trainable_variables
regularization_losses
layer_regularization_losses

 layers
	variables
trainable_variables
!metrics
US
VARIABLE_VALUEmy_model/dense_5/kernel(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEmy_model/dense_5/bias&dense2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

"non_trainable_variables
regularization_losses
#layer_regularization_losses

$layers
	variables
trainable_variables
%metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

&0
 
 
 
 
 
 
 
 
x
	'total
	(count
)
_fn_kwargs
*regularization_losses
+	variables
,trainable_variables
-	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 

'0
(1
 

.non_trainable_variables
*regularization_losses
/layer_regularization_losses

0layers
+	variables
,trainable_variables
1metrics

'0
(1
 
 
 
xv
VARIABLE_VALUEAdam/my_model/dense_4/kernel/mDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model/dense_4/bias/mBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/my_model/dense_5/kernel/mDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model/dense_5/bias/mBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/my_model/dense_4/kernel/vDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model/dense_4/bias/vBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/my_model/dense_5/kernel/vDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/my_model/dense_5/bias/vBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|
serving_default_input_1Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model/dense_4/kernelmy_model/dense_4/biasmy_model/dense_5/kernelmy_model/dense_5/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
config_proto

CPU

GPU2*0J 8*,
f'R%
#__inference_signature_wrapper_13134
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Á
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+my_model/dense_4/kernel/Read/ReadVariableOp)my_model/dense_4/bias/Read/ReadVariableOp+my_model/dense_5/kernel/Read/ReadVariableOp)my_model/dense_5/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp2Adam/my_model/dense_4/kernel/m/Read/ReadVariableOp0Adam/my_model/dense_4/bias/m/Read/ReadVariableOp2Adam/my_model/dense_5/kernel/m/Read/ReadVariableOp0Adam/my_model/dense_5/bias/m/Read/ReadVariableOp2Adam/my_model/dense_4/kernel/v/Read/ReadVariableOp0Adam/my_model/dense_4/bias/v/Read/ReadVariableOp2Adam/my_model/dense_5/kernel/v/Read/ReadVariableOp0Adam/my_model/dense_5/bias/v/Read/ReadVariableOpConst* 
Tin
2	*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*'
f"R 
__inference__traced_save_13250
À
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model/dense_4/kernelmy_model/dense_4/biasmy_model/dense_5/kernelmy_model/dense_5/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/my_model/dense_4/kernel/mAdam/my_model/dense_4/bias/mAdam/my_model/dense_5/kernel/mAdam/my_model/dense_5/bias/mAdam/my_model/dense_4/kernel/vAdam/my_model/dense_4/bias/vAdam/my_model/dense_5/kernel/vAdam/my_model/dense_5/bias/v*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_restore_13319÷ì
Å	
Û
B__inference_dense_4_layer_call_and_return_conditional_losses_13068

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
Å	
Û
B__inference_dense_4_layer_call_and_return_conditional_losses_13145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs

ò
(__inference_my_model_layer_call_fn_13116
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCallË
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
config_proto

CPU

GPU2*0J 8*L
fGRE
C__inference_my_model_layer_call_and_return_conditional_losses_131062
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
½
×
C__inference_my_model_layer_call_and_return_conditional_losses_13106
input_1*
&dense_4_statefulpartitionedcall_args_1*
&dense_4_statefulpartitionedcall_args_2*
&dense_5_statefulpartitionedcall_args_1*
&dense_5_statefulpartitionedcall_args_2
identity¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¨
dense_4/StatefulPartitionedCallStatefulPartitionedCallinput_1&dense_4_statefulpartitionedcall_args_1&dense_4_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_130682!
dense_4/StatefulPartitionedCall®
dense_4/IdentityIdentity(dense_4/StatefulPartitionedCall:output:0 ^dense_4/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_4/Identitya
ReluReludense_4/Identity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Relu³
dense_5/StatefulPartitionedCallStatefulPartitionedCallRelu:activations:0&dense_5_statefulpartitionedcall_args_1&dense_5_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_130922!
dense_5/StatefulPartitionedCall®
dense_5/IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
dense_5/Identity±
IdentityIdentitydense_5/Identity:output:0 ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:' #
!
_user_specified_name	input_1
ñ
¨
'__inference_dense_5_layer_call_fn_13169

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_130922
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ç
Û
B__inference_dense_5_layer_call_and_return_conditional_losses_13092

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
º
¼
 __inference__wrapped_model_13053
input_13
/my_model_dense_4_matmul_readvariableop_resource4
0my_model_dense_4_biasadd_readvariableop_resource3
/my_model_dense_5_matmul_readvariableop_resource4
0my_model_dense_5_biasadd_readvariableop_resource
identity¢'my_model/dense_4/BiasAdd/ReadVariableOp¢&my_model/dense_4/MatMul/ReadVariableOp¢'my_model/dense_5/BiasAdd/ReadVariableOp¢&my_model/dense_5/MatMul/ReadVariableOpÁ
&my_model/dense_4/MatMul/ReadVariableOpReadVariableOp/my_model_dense_4_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02(
&my_model/dense_4/MatMul/ReadVariableOp§
my_model/dense_4/MatMulMatMulinput_1.my_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
my_model/dense_4/MatMul¿
'my_model/dense_4/BiasAdd/ReadVariableOpReadVariableOp0my_model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'my_model/dense_4/BiasAdd/ReadVariableOpÅ
my_model/dense_4/BiasAddBiasAdd!my_model/dense_4/MatMul:product:0/my_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
my_model/dense_4/BiasAdd
my_model/dense_4/ReluRelu!my_model/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
my_model/dense_4/Relu}
my_model/ReluRelu#my_model/dense_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
my_model/ReluÀ
&my_model/dense_5/MatMul/ReadVariableOpReadVariableOp/my_model_dense_5_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02(
&my_model/dense_5/MatMul/ReadVariableOp»
my_model/dense_5/MatMulMatMulmy_model/Relu:activations:0.my_model/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
my_model/dense_5/MatMul¿
'my_model/dense_5/BiasAdd/ReadVariableOpReadVariableOp0my_model_dense_5_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02)
'my_model/dense_5/BiasAdd/ReadVariableOpÅ
my_model/dense_5/BiasAddBiasAdd!my_model/dense_5/MatMul:product:0/my_model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
my_model/dense_5/BiasAdd
IdentityIdentity!my_model/dense_5/BiasAdd:output:0(^my_model/dense_4/BiasAdd/ReadVariableOp'^my_model/dense_4/MatMul/ReadVariableOp(^my_model/dense_5/BiasAdd/ReadVariableOp'^my_model/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2R
'my_model/dense_4/BiasAdd/ReadVariableOp'my_model/dense_4/BiasAdd/ReadVariableOp2P
&my_model/dense_4/MatMul/ReadVariableOp&my_model/dense_4/MatMul/ReadVariableOp2R
'my_model/dense_5/BiasAdd/ReadVariableOp'my_model/dense_5/BiasAdd/ReadVariableOp2P
&my_model/dense_5/MatMul/ReadVariableOp&my_model/dense_5/MatMul/ReadVariableOp:' #
!
_user_specified_name	input_1
á
í
#__inference_signature_wrapper_13134
input_1"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinput_1statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*-
config_proto

CPU

GPU2*0J 8*)
f$R"
 __inference__wrapped_model_130532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_1
ÑR
ß

!__inference__traced_restore_13319
file_prefix,
(assignvariableop_my_model_dense_4_kernel,
(assignvariableop_1_my_model_dense_4_bias.
*assignvariableop_2_my_model_dense_5_kernel,
(assignvariableop_3_my_model_dense_5_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate
assignvariableop_9_total
assignvariableop_10_count6
2assignvariableop_11_adam_my_model_dense_4_kernel_m4
0assignvariableop_12_adam_my_model_dense_4_bias_m6
2assignvariableop_13_adam_my_model_dense_5_kernel_m4
0assignvariableop_14_adam_my_model_dense_5_bias_m6
2assignvariableop_15_adam_my_model_dense_4_kernel_v4
0assignvariableop_16_adam_my_model_dense_4_bias_v6
2assignvariableop_17_adam_my_model_dense_5_kernel_v4
0assignvariableop_18_adam_my_model_dense_5_bias_v
identity_20¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¢	RestoreV2¢RestoreV2_1¦	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B¥B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names´
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp(assignvariableop_my_model_dense_4_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1
AssignVariableOp_1AssignVariableOp(assignvariableop_1_my_model_dense_4_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2 
AssignVariableOp_2AssignVariableOp*assignvariableop_2_my_model_dense_5_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3
AssignVariableOp_3AssignVariableOp(assignvariableop_3_my_model_dense_5_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0	*
_output_shapes
:2

Identity_4
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11«
AssignVariableOp_11AssignVariableOp2assignvariableop_11_adam_my_model_dense_4_kernel_mIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12©
AssignVariableOp_12AssignVariableOp0assignvariableop_12_adam_my_model_dense_4_bias_mIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13«
AssignVariableOp_13AssignVariableOp2assignvariableop_13_adam_my_model_dense_5_kernel_mIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14©
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_my_model_dense_5_bias_mIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15«
AssignVariableOp_15AssignVariableOp2assignvariableop_15_adam_my_model_dense_4_kernel_vIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16©
AssignVariableOp_16AssignVariableOp0assignvariableop_16_adam_my_model_dense_4_bias_vIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17«
AssignVariableOp_17AssignVariableOp2assignvariableop_17_adam_my_model_dense_5_kernel_vIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18©
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_my_model_dense_5_bias_vIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18¨
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesÄ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_19Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_19
Identity_20IdentityIdentity_19:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_20"#
identity_20Identity_20:output:0*a
_input_shapesP
N: :::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
ç
Û
B__inference_dense_5_layer_call_and_return_conditional_losses_13162

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2	
BiasAdd
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs
æ/
ù
__inference__traced_save_13250
file_prefix6
2savev2_my_model_dense_4_kernel_read_readvariableop4
0savev2_my_model_dense_4_bias_read_readvariableop6
2savev2_my_model_dense_5_kernel_read_readvariableop4
0savev2_my_model_dense_5_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop=
9savev2_adam_my_model_dense_4_kernel_m_read_readvariableop;
7savev2_adam_my_model_dense_4_bias_m_read_readvariableop=
9savev2_adam_my_model_dense_5_kernel_m_read_readvariableop;
7savev2_adam_my_model_dense_5_bias_m_read_readvariableop=
9savev2_adam_my_model_dense_4_kernel_v_read_readvariableop;
7savev2_adam_my_model_dense_4_bias_v_read_readvariableop=
9savev2_adam_my_model_dense_5_kernel_v_read_readvariableop;
7savev2_adam_my_model_dense_5_bias_v_read_readvariableop
savev2_1_const

identity_1¢MergeV2Checkpoints¢SaveV2¢SaveV2_1¥
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_09b365dbc5ab400685dcead8085de8ba/part2
StringJoin/inputs_1

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename 	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B¥B(dense1/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense1/bias/.ATTRIBUTES/VARIABLE_VALUEB(dense2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&dense2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBDdense1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBDdense2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBdense2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names®
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesê
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_my_model_dense_4_kernel_read_readvariableop0savev2_my_model_dense_4_bias_read_readvariableop2savev2_my_model_dense_5_kernel_read_readvariableop0savev2_my_model_dense_5_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop9savev2_adam_my_model_dense_4_kernel_m_read_readvariableop7savev2_adam_my_model_dense_4_bias_m_read_readvariableop9savev2_adam_my_model_dense_5_kernel_m_read_readvariableop7savev2_adam_my_model_dense_5_bias_m_read_readvariableop9savev2_adam_my_model_dense_4_kernel_v_read_readvariableop7savev2_adam_my_model_dense_4_bias_v_read_readvariableop9savev2_adam_my_model_dense_5_kernel_v_read_readvariableop7savev2_adam_my_model_dense_5_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *!
dtypes
2	2
SaveV2
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard¬
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1¢
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slicesÏ
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1ã
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¬
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesw
u: :	@:@:@
:
: : : : : : : :	@:@:@
:
:	@:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
ò
¨
'__inference_dense_4_layer_call_fn_13152

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*-
config_proto

CPU

GPU2*0J 8*K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_130682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs"¯L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¬
serving_default
<
input_11
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:L
ù

dense1

dense2
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
:__call__
;_default_save_signature
*<&call_and_return_all_conditional_losses"
_tf_keras_modelü{"class_name": "MyModel", "name": "my_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}}, "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ò

	kernel

bias
regularization_losses
	variables
trainable_variables
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
ó

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}}

iter

beta_1

beta_2
	decay
learning_rate	m2
m3m4m5	v6
v7v8v9"
	optimizer
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
·
non_trainable_variables
regularization_losses
layer_regularization_losses

layers
	variables
trainable_variables
metrics
:__call__
;_default_save_signature
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
,
Aserving_default"
signature_map
*:(	@2my_model/dense_4/kernel
#:!@2my_model/dense_4/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper

non_trainable_variables
regularization_losses
layer_regularization_losses

 layers
	variables
trainable_variables
!metrics
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
):'@
2my_model/dense_5/kernel
#:!
2my_model/dense_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

"non_trainable_variables
regularization_losses
#layer_regularization_losses

$layers
	variables
trainable_variables
%metrics
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

	'total
	(count
)
_fn_kwargs
*regularization_losses
+	variables
,trainable_variables
-	keras_api
B__call__
*C&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper

.non_trainable_variables
*regularization_losses
/layer_regularization_losses

0layers
+	variables
,trainable_variables
1metrics
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/:-	@2Adam/my_model/dense_4/kernel/m
(:&@2Adam/my_model/dense_4/bias/m
.:,@
2Adam/my_model/dense_5/kernel/m
(:&
2Adam/my_model/dense_5/bias/m
/:-	@2Adam/my_model/dense_4/kernel/v
(:&@2Adam/my_model/dense_4/bias/v
.:,@
2Adam/my_model/dense_5/kernel/v
(:&
2Adam/my_model/dense_5/bias/v
ý2ú
(__inference_my_model_layer_call_fn_13116Í
²
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ß2Ü
 __inference__wrapped_model_13053·
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
2
C__inference_my_model_layer_call_and_return_conditional_losses_13106Í
²
FullArgSpec#
args
jself
jinput_tensor
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
Ñ2Î
'__inference_dense_4_layer_call_fn_13152¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_4_layer_call_and_return_conditional_losses_13145¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ñ2Î
'__inference_dense_5_layer_call_fn_13169¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_5_layer_call_and_return_conditional_losses_13162¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2B0
#__inference_signature_wrapper_13134input_1
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
Ì2ÉÆ
½²¹
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsª

trainingp 
annotationsª *
 
 __inference__wrapped_model_13053n	
1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
£
B__inference_dense_4_layer_call_and_return_conditional_losses_13145]	
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 {
'__inference_dense_4_layer_call_fn_13152P	
0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¢
B__inference_dense_5_layer_call_and_return_conditional_losses_13162\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 z
'__inference_dense_5_layer_call_fn_13169O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
§
C__inference_my_model_layer_call_and_return_conditional_losses_13106`	
1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
(__inference_my_model_layer_call_fn_13116S	
1¢.
'¢$
"
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
 
#__inference_signature_wrapper_13134y	
<¢9
¢ 
2ª/
-
input_1"
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ
