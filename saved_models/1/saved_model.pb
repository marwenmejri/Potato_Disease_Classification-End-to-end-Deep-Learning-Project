¢3
Ú
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	

ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
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
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.6.12v2.6.0-101-g3aa40c3ce9d8À0

conv2d_90/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_90/kernel
}
$conv2d_90/kernel/Read/ReadVariableOpReadVariableOpconv2d_90/kernel*&
_output_shapes
: *
dtype0
t
conv2d_90/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_90/bias
m
"conv2d_90/bias/Read/ReadVariableOpReadVariableOpconv2d_90/bias*
_output_shapes
: *
dtype0

conv2d_91/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*!
shared_nameconv2d_91/kernel
}
$conv2d_91/kernel/Read/ReadVariableOpReadVariableOpconv2d_91/kernel*&
_output_shapes
: @*
dtype0
t
conv2d_91/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_91/bias
m
"conv2d_91/bias/Read/ReadVariableOpReadVariableOpconv2d_91/bias*
_output_shapes
:@*
dtype0

conv2d_92/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_92/kernel
}
$conv2d_92/kernel/Read/ReadVariableOpReadVariableOpconv2d_92/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_92/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_92/bias
m
"conv2d_92/bias/Read/ReadVariableOpReadVariableOpconv2d_92/bias*
_output_shapes
:@*
dtype0

conv2d_93/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_93/kernel
}
$conv2d_93/kernel/Read/ReadVariableOpReadVariableOpconv2d_93/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_93/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_93/bias
m
"conv2d_93/bias/Read/ReadVariableOpReadVariableOpconv2d_93/bias*
_output_shapes
:@*
dtype0

conv2d_94/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_94/kernel
}
$conv2d_94/kernel/Read/ReadVariableOpReadVariableOpconv2d_94/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_94/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_94/bias
m
"conv2d_94/bias/Read/ReadVariableOpReadVariableOpconv2d_94/bias*
_output_shapes
:@*
dtype0

conv2d_95/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_95/kernel
}
$conv2d_95/kernel/Read/ReadVariableOpReadVariableOpconv2d_95/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_95/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_95/bias
m
"conv2d_95/bias/Read/ReadVariableOpReadVariableOpconv2d_95/bias*
_output_shapes
:@*
dtype0
{
dense_30/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@* 
shared_namedense_30/kernel
t
#dense_30/kernel/Read/ReadVariableOpReadVariableOpdense_30/kernel*
_output_shapes
:	@*
dtype0
r
dense_30/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_30/bias
k
!dense_30/bias/Read/ReadVariableOpReadVariableOpdense_30/bias*
_output_shapes
:@*
dtype0
z
dense_31/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@* 
shared_namedense_31/kernel
s
#dense_31/kernel/Read/ReadVariableOpReadVariableOpdense_31/kernel*
_output_shapes

:@*
dtype0
r
dense_31/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_31/bias
k
!dense_31/bias/Read/ReadVariableOpReadVariableOpdense_31/bias*
_output_shapes
:*
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
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
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
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_90/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_90/kernel/m

+Adam/conv2d_90/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/kernel/m*&
_output_shapes
: *
dtype0

Adam/conv2d_90/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_90/bias/m
{
)Adam/conv2d_90/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/bias/m*
_output_shapes
: *
dtype0

Adam/conv2d_91/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_91/kernel/m

+Adam/conv2d_91/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_91/kernel/m*&
_output_shapes
: @*
dtype0

Adam/conv2d_91/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_91/bias/m
{
)Adam/conv2d_91/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_91/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_92/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_92/kernel/m

+Adam/conv2d_92/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_92/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_92/bias/m
{
)Adam/conv2d_92/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_93/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_93/kernel/m

+Adam/conv2d_93/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_93/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_93/bias/m
{
)Adam/conv2d_93/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_94/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_94/kernel/m

+Adam/conv2d_94/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_94/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_94/bias/m
{
)Adam/conv2d_94/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_95/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_95/kernel/m

+Adam/conv2d_95/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_95/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_95/bias/m
{
)Adam/conv2d_95/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_30/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_30/kernel/m

*Adam/dense_30/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/m*
_output_shapes
:	@*
dtype0

Adam/dense_30/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_30/bias/m
y
(Adam/dense_30/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/m*
_output_shapes
:@*
dtype0

Adam/dense_31/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_31/kernel/m

*Adam/dense_31/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/m*
_output_shapes

:@*
dtype0

Adam/dense_31/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/m
y
(Adam/dense_31/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_90/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_90/kernel/v

+Adam/conv2d_90/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/kernel/v*&
_output_shapes
: *
dtype0

Adam/conv2d_90/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_90/bias/v
{
)Adam/conv2d_90/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_90/bias/v*
_output_shapes
: *
dtype0

Adam/conv2d_91/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameAdam/conv2d_91/kernel/v

+Adam/conv2d_91/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_91/kernel/v*&
_output_shapes
: @*
dtype0

Adam/conv2d_91/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_91/bias/v
{
)Adam/conv2d_91/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_91/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_92/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_92/kernel/v

+Adam/conv2d_92/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_92/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_92/bias/v
{
)Adam/conv2d_92/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_92/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_93/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_93/kernel/v

+Adam/conv2d_93/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_93/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_93/bias/v
{
)Adam/conv2d_93/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_93/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_94/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_94/kernel/v

+Adam/conv2d_94/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_94/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_94/bias/v
{
)Adam/conv2d_94/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_94/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_95/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_95/kernel/v

+Adam/conv2d_95/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_95/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_95/bias/v
{
)Adam/conv2d_95/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_95/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_30/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@*'
shared_nameAdam/dense_30/kernel/v

*Adam/dense_30/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/kernel/v*
_output_shapes
:	@*
dtype0

Adam/dense_30/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/dense_30/bias/v
y
(Adam/dense_30/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_30/bias/v*
_output_shapes
:@*
dtype0

Adam/dense_31/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*'
shared_nameAdam/dense_31/kernel/v

*Adam/dense_31/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/kernel/v*
_output_shapes

:@*
dtype0

Adam/dense_31/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_31/bias/v
y
(Adam/dense_31/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_31/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ér
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¤r
valuerBr Br
¥
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
l
layer-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api
l
layer-0
layer-1
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
R
4trainable_variables
5	variables
6regularization_losses
7	keras_api
h

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
R
>trainable_variables
?	variables
@regularization_losses
A	keras_api
h

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
R
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
R
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
h

Vkernel
Wbias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
R
\trainable_variables
]	variables
^regularization_losses
_	keras_api
R
`trainable_variables
a	variables
bregularization_losses
c	keras_api
h

dkernel
ebias
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
h

jkernel
kbias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api

piter

qbeta_1

rbeta_2
	sdecay
tlearning_rate$m%m.m/m8m9mBmCmLmMmVmWmdmemjmkm$v%v.v/v8v9vBvCvLvMvVvWvdvevjv kv¡
v
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15
v
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15
 
­
trainable_variables
	variables
ulayer_regularization_losses
regularization_losses
vmetrics
wlayer_metrics

xlayers
ynon_trainable_variables
 
R
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
T
~trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
²
trainable_variables
	variables
 layer_regularization_losses
regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
a
	_rng
trainable_variables
	variables
regularization_losses
	keras_api
a
	_rng
trainable_variables
	variables
regularization_losses
	keras_api
 
 
 
²
 trainable_variables
!	variables
 layer_regularization_losses
"regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
\Z
VARIABLE_VALUEconv2d_90/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_90/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
²
&trainable_variables
'	variables
 layer_regularization_losses
(regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
 
 
 
²
*trainable_variables
+	variables
 layer_regularization_losses
,regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
\Z
VARIABLE_VALUEconv2d_91/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_91/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
²
0trainable_variables
1	variables
  layer_regularization_losses
2regularization_losses
¡metrics
¢layer_metrics
£layers
¤non_trainable_variables
 
 
 
²
4trainable_variables
5	variables
 ¥layer_regularization_losses
6regularization_losses
¦metrics
§layer_metrics
¨layers
©non_trainable_variables
\Z
VARIABLE_VALUEconv2d_92/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_92/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91

80
91
 
²
:trainable_variables
;	variables
 ªlayer_regularization_losses
<regularization_losses
«metrics
¬layer_metrics
­layers
®non_trainable_variables
 
 
 
²
>trainable_variables
?	variables
 ¯layer_regularization_losses
@regularization_losses
°metrics
±layer_metrics
²layers
³non_trainable_variables
\Z
VARIABLE_VALUEconv2d_93/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_93/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
 
²
Dtrainable_variables
E	variables
 ´layer_regularization_losses
Fregularization_losses
µmetrics
¶layer_metrics
·layers
¸non_trainable_variables
 
 
 
²
Htrainable_variables
I	variables
 ¹layer_regularization_losses
Jregularization_losses
ºmetrics
»layer_metrics
¼layers
½non_trainable_variables
\Z
VARIABLE_VALUEconv2d_94/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_94/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
²
Ntrainable_variables
O	variables
 ¾layer_regularization_losses
Pregularization_losses
¿metrics
Àlayer_metrics
Álayers
Ânon_trainable_variables
 
 
 
²
Rtrainable_variables
S	variables
 Ãlayer_regularization_losses
Tregularization_losses
Ämetrics
Ålayer_metrics
Ælayers
Çnon_trainable_variables
\Z
VARIABLE_VALUEconv2d_95/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_95/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

V0
W1
 
²
Xtrainable_variables
Y	variables
 Èlayer_regularization_losses
Zregularization_losses
Émetrics
Êlayer_metrics
Ëlayers
Ìnon_trainable_variables
 
 
 
²
\trainable_variables
]	variables
 Ílayer_regularization_losses
^regularization_losses
Îmetrics
Ïlayer_metrics
Ðlayers
Ñnon_trainable_variables
 
 
 
²
`trainable_variables
a	variables
 Òlayer_regularization_losses
bregularization_losses
Ómetrics
Ôlayer_metrics
Õlayers
Önon_trainable_variables
[Y
VARIABLE_VALUEdense_30/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_30/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

d0
e1
 
²
ftrainable_variables
g	variables
 ×layer_regularization_losses
hregularization_losses
Ømetrics
Ùlayer_metrics
Úlayers
Ûnon_trainable_variables
[Y
VARIABLE_VALUEdense_31/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_31/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
²
ltrainable_variables
m	variables
 Ülayer_regularization_losses
nregularization_losses
Ýmetrics
Þlayer_metrics
ßlayers
ànon_trainable_variables
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

á0
â1
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
²
ztrainable_variables
{	variables
 ãlayer_regularization_losses
|regularization_losses
ämetrics
ålayer_metrics
ælayers
çnon_trainable_variables
 
 
 
³
~trainable_variables
	variables
 èlayer_regularization_losses
regularization_losses
émetrics
êlayer_metrics
ëlayers
ìnon_trainable_variables
 
 
 

0
1
 

í
_state_var
 
 
 
µ
trainable_variables
	variables
 îlayer_regularization_losses
regularization_losses
ïmetrics
ðlayer_metrics
ñlayers
ònon_trainable_variables

ó
_state_var
 
 
 
µ
trainable_variables
	variables
 ôlayer_regularization_losses
regularization_losses
õmetrics
ölayer_metrics
÷layers
ønon_trainable_variables
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ùtotal

úcount
û	variables
ü	keras_api
I

ýtotal

þcount
ÿ
_fn_kwargs
	variables
	keras_api
 
 
 
 
 
 
 
 
 
 
XV
VARIABLE_VALUEVariable:layer-1/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
ZX
VARIABLE_VALUE
Variable_1:layer-1/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ù0
ú1

û	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ý0
þ1

	variables
}
VARIABLE_VALUEAdam/conv2d_90/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_90/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_91/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_91/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_92/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_92/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_93/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_93/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_94/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_94/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_95/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_95/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_90/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_90/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_91/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_91/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_92/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_92/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_93/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_93/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_94/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_94/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_95/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_95/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_30/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_30/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_31/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_31/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

 serving_default_sequential_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
ç
StatefulPartitionedCallStatefulPartitionedCall serving_default_sequential_inputconv2d_90/kernelconv2d_90/biasconv2d_91/kernelconv2d_91/biasconv2d_92/kernelconv2d_92/biasconv2d_93/kernelconv2d_93/biasconv2d_94/kernelconv2d_94/biasconv2d_95/kernelconv2d_95/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_162187
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ñ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_90/kernel/Read/ReadVariableOp"conv2d_90/bias/Read/ReadVariableOp$conv2d_91/kernel/Read/ReadVariableOp"conv2d_91/bias/Read/ReadVariableOp$conv2d_92/kernel/Read/ReadVariableOp"conv2d_92/bias/Read/ReadVariableOp$conv2d_93/kernel/Read/ReadVariableOp"conv2d_93/bias/Read/ReadVariableOp$conv2d_94/kernel/Read/ReadVariableOp"conv2d_94/bias/Read/ReadVariableOp$conv2d_95/kernel/Read/ReadVariableOp"conv2d_95/bias/Read/ReadVariableOp#dense_30/kernel/Read/ReadVariableOp!dense_30/bias/Read/ReadVariableOp#dense_31/kernel/Read/ReadVariableOp!dense_31/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_90/kernel/m/Read/ReadVariableOp)Adam/conv2d_90/bias/m/Read/ReadVariableOp+Adam/conv2d_91/kernel/m/Read/ReadVariableOp)Adam/conv2d_91/bias/m/Read/ReadVariableOp+Adam/conv2d_92/kernel/m/Read/ReadVariableOp)Adam/conv2d_92/bias/m/Read/ReadVariableOp+Adam/conv2d_93/kernel/m/Read/ReadVariableOp)Adam/conv2d_93/bias/m/Read/ReadVariableOp+Adam/conv2d_94/kernel/m/Read/ReadVariableOp)Adam/conv2d_94/bias/m/Read/ReadVariableOp+Adam/conv2d_95/kernel/m/Read/ReadVariableOp)Adam/conv2d_95/bias/m/Read/ReadVariableOp*Adam/dense_30/kernel/m/Read/ReadVariableOp(Adam/dense_30/bias/m/Read/ReadVariableOp*Adam/dense_31/kernel/m/Read/ReadVariableOp(Adam/dense_31/bias/m/Read/ReadVariableOp+Adam/conv2d_90/kernel/v/Read/ReadVariableOp)Adam/conv2d_90/bias/v/Read/ReadVariableOp+Adam/conv2d_91/kernel/v/Read/ReadVariableOp)Adam/conv2d_91/bias/v/Read/ReadVariableOp+Adam/conv2d_92/kernel/v/Read/ReadVariableOp)Adam/conv2d_92/bias/v/Read/ReadVariableOp+Adam/conv2d_93/kernel/v/Read/ReadVariableOp)Adam/conv2d_93/bias/v/Read/ReadVariableOp+Adam/conv2d_94/kernel/v/Read/ReadVariableOp)Adam/conv2d_94/bias/v/Read/ReadVariableOp+Adam/conv2d_95/kernel/v/Read/ReadVariableOp)Adam/conv2d_95/bias/v/Read/ReadVariableOp*Adam/dense_30/kernel/v/Read/ReadVariableOp(Adam/dense_30/bias/v/Read/ReadVariableOp*Adam/dense_31/kernel/v/Read/ReadVariableOp(Adam/dense_31/bias/v/Read/ReadVariableOpConst*H
TinA
?2=			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_164861
Ð
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_90/kernelconv2d_90/biasconv2d_91/kernelconv2d_91/biasconv2d_92/kernelconv2d_92/biasconv2d_93/kernelconv2d_93/biasconv2d_94/kernelconv2d_94/biasconv2d_95/kernelconv2d_95/biasdense_30/kerneldense_30/biasdense_31/kerneldense_31/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1totalcounttotal_1count_1Adam/conv2d_90/kernel/mAdam/conv2d_90/bias/mAdam/conv2d_91/kernel/mAdam/conv2d_91/bias/mAdam/conv2d_92/kernel/mAdam/conv2d_92/bias/mAdam/conv2d_93/kernel/mAdam/conv2d_93/bias/mAdam/conv2d_94/kernel/mAdam/conv2d_94/bias/mAdam/conv2d_95/kernel/mAdam/conv2d_95/bias/mAdam/dense_30/kernel/mAdam/dense_30/bias/mAdam/dense_31/kernel/mAdam/dense_31/bias/mAdam/conv2d_90/kernel/vAdam/conv2d_90/bias/vAdam/conv2d_91/kernel/vAdam/conv2d_91/bias/vAdam/conv2d_92/kernel/vAdam/conv2d_92/bias/vAdam/conv2d_93/kernel/vAdam/conv2d_93/bias/vAdam/conv2d_94/kernel/vAdam/conv2d_94/bias/vAdam/conv2d_95/kernel/vAdam/conv2d_95/bias/vAdam/dense_30/kernel/vAdam/dense_30/bias/vAdam/dense_31/kernel/vAdam/dense_31/bias/v*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_165048Èõ-
¢
d
H__inference_sequential_1_layer_call_and_return_conditional_losses_163795

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
I
-__inference_sequential_1_layer_call_fn_163197

inputs
identityË
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1607542
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

b
F__inference_sequential_layer_call_and_return_conditional_losses_163177

inputs
identitym
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulinputsrescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/add}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeÜ
resizing/resize/ResizeBilinearResizeBilinearrescaling_1/add:z:0resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2 
resizing/resize/ResizeBilinear
IdentityIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
M
1__inference_max_pooling2d_94_layer_call_fn_164210

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_1613782
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


*__inference_conv2d_95_layer_call_fn_164229

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1613912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

µ
*stateless_random_flip_up_down_false_164499[
Wstateless_random_flip_up_down_identity_stateless_random_flip_up_down_control_dependency*
&stateless_random_flip_up_down_identityì
&stateless_random_flip_up_down/IdentityIdentityWstateless_random_flip_up_down_identity_stateless_random_flip_up_down_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_up_down/Identity"Y
&stateless_random_flip_up_down_identity/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

ö
D__inference_dense_30_layer_call_and_return_conditional_losses_164291

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
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
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
I
-__inference_sequential_1_layer_call_fn_163220

inputs
identityÐ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1612632
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
À
E
)__inference_resizing_layer_call_fn_164329

inputs
identity¿
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_1606712
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_161207

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_161163

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
G
+__inference_sequential_layer_call_fn_163111

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1618532
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à
Ä
K__inference_random_rotation_layer_call_and_return_conditional_losses_164651

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimsQ
ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¸
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¯
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concate
transform/ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_valueÝ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3ExpandDims:output:0rotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3¦
SqueezeSqueeze9transform/ImageProjectiveTransformV3:transformed_images:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2	
Squeezep
IdentityIdentitySqueeze:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_94_layer_call_and_return_conditional_losses_161368

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

m
F__inference_sequential_layer_call_and_return_conditional_losses_163167
rescaling_1_input
identitym
CastCastrescaling_1_input*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
Castm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulCast:y:0rescaling_1/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/add
resizing/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
resizing/resize/ExpandDims/dim¿
resizing/resize/ExpandDims
ExpandDimsrescaling_1/add:z:0'resizing/resize/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
resizing/resize/ExpandDims}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeã
resizing/resize/ResizeBilinearResizeBilinear#resizing/resize/ExpandDims:output:0resizing/resize/size:output:0*
T0*(
_output_shapes
:*
half_pixel_centers(2 
resizing/resize/ResizeBilinear´
resizing/resize/SqueezeSqueeze/resizing/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 2
resizing/resize/Squeezeq
IdentityIdentity resizing/resize/Squeeze:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
+
_user_specified_namerescaling_1_input
«
Á
-stateless_random_flip_left_right_false_160968a
]stateless_random_flip_left_right_identity_stateless_random_flip_left_right_control_dependency-
)stateless_random_flip_left_right_identityø
)stateless_random_flip_left_right/IdentityIdentity]stateless_random_flip_left_right_identity_stateless_random_flip_left_right_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateless_random_flip_left_right/Identity"_
)stateless_random_flip_left_right_identity2stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
h
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_164255

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

b
F__inference_sequential_layer_call_and_return_conditional_losses_161258

inputs
identitym
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulinputsrescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/add}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeÜ
resizing/resize/ResizeBilinearResizeBilinearrescaling_1/add:z:0resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2 
resizing/resize/ResizeBilinear
IdentityIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_161401

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è
G
+__inference_sequential_layer_call_fn_163106

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1612582
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ç°

H__inference_sequential_1_layer_call_and_return_conditional_losses_164020

inputsK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	
identity¢4random_flip/stateful_uniform_full_int/RngReadAndSkip¢6random_flip/stateful_uniform_full_int_1/RngReadAndSkip¢[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter¢/random_rotation/stateful_uniform/RngReadAndSkip¤
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_flip/stateful_uniform_full_int/shape¤
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+random_flip/stateful_uniform_full_int/Constí
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2,
*random_flip/stateful_uniform_full_int/Prod
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2.
,random_flip/stateful_uniform_full_int/Cast/xÉ
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int/Cast_1Â
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:26
4random_flip/stateful_uniform_full_int/RngReadAndSkipÀ
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip/stateful_uniform_full_int/strided_slice/stackÄ
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Ä
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Ì
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask25
3random_flip/stateful_uniform_full_int/strided_sliceØ
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02/
-random_flip/stateful_uniform_full_int/BitcastÄ
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice_1/stackÈ
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1È
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Ä
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:27
5random_flip/stateful_uniform_full_int/strided_slice_1Þ
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int/Bitcast_1
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_flip/stateful_uniform_full_int/algö
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2'
%random_flip/stateful_uniform_full_intz
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like±
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
random_flip/strided_slice/stack
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!random_flip/strided_slice/stack_1
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!random_flip/strided_slice/stack_2Ð
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_sliceí
?random_flip/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?random_flip/stateless_random_flip_left_right/control_dependencyà
2random_flip/stateless_random_flip_left_right/ShapeShapeHrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:24
2random_flip/stateless_random_flip_left_right/ShapeÎ
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@random_flip/stateless_random_flip_left_right/strided_slice/stackÒ
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Ò
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2ð
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:random_flip/stateless_random_flip_left_right/strided_slice
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2M
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max®
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0* 
_output_shapes
::2d
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterÐ
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2]
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0arandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2æ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulè
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniform¾
<random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<random_flip/stateless_random_flip_left_right/Reshape/shape/1¾
<random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2>
<random_flip/stateless_random_flip_left_right/Reshape/shape/2¾
<random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2>
<random_flip/stateless_random_flip_left_right/Reshape/shape/3È
:random_flip/stateless_random_flip_left_right/Reshape/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:random_flip/stateless_random_flip_left_right/Reshape/shapeÁ
4random_flip/stateless_random_flip_left_right/ReshapeReshapeIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Crandom_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4random_flip/stateless_random_flip_left_right/Reshapeê
2random_flip/stateless_random_flip_left_right/RoundRound=random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_left_right/RoundÄ
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateless_random_flip_left_right/ReverseV2/axisÉ
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0Drandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6random_flip/stateless_random_flip_left_right/ReverseV2 
0random_flip/stateless_random_flip_left_right/mulMul6random_flip/stateless_random_flip_left_right/Round:y:0?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0random_flip/stateless_random_flip_left_right/mul­
2random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2random_flip/stateless_random_flip_left_right/sub/x
0random_flip/stateless_random_flip_left_right/subSub;random_flip/stateless_random_flip_left_right/sub/x:output:06random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0random_flip/stateless_random_flip_left_right/sub«
2random_flip/stateless_random_flip_left_right/mul_1Mul4random_flip/stateless_random_flip_left_right/sub:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_left_right/mul_1
0random_flip/stateless_random_flip_left_right/addAddV24random_flip/stateless_random_flip_left_right/mul:z:06random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0random_flip/stateless_random_flip_left_right/add¨
-random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_flip/stateful_uniform_full_int_1/shape¨
-random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_flip/stateful_uniform_full_int_1/Constõ
,random_flip/stateful_uniform_full_int_1/ProdProd6random_flip/stateful_uniform_full_int_1/shape:output:06random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int_1/Prod¢
.random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip/stateful_uniform_full_int_1/Cast/xÏ
.random_flip/stateful_uniform_full_int_1/Cast_1Cast5random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.random_flip/stateful_uniform_full_int_1/Cast_1
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource7random_flip/stateful_uniform_full_int_1/Cast/x:output:02random_flip/stateful_uniform_full_int_1/Cast_1:y:05^random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:28
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipÄ
;random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;random_flip/stateful_uniform_full_int_1/strided_slice/stackÈ
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1È
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Ø
5random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask27
5random_flip/stateful_uniform_full_int_1/strided_sliceÞ
/random_flip/stateful_uniform_full_int_1/BitcastBitcast>random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int_1/BitcastÈ
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackÌ
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Ì
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Ð
7random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Frandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:29
7random_flip/stateful_uniform_full_int_1/strided_slice_1ä
1random_flip/stateful_uniform_full_int_1/Bitcast_1Bitcast@random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type023
1random_flip/stateful_uniform_full_int_1/Bitcast_1
+random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_flip/stateful_uniform_full_int_1/alg
'random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV26random_flip/stateful_uniform_full_int_1/shape:output:0:random_flip/stateful_uniform_full_int_1/Bitcast_1:output:08random_flip/stateful_uniform_full_int_1/Bitcast:output:04random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	2)
'random_flip/stateful_uniform_full_int_1~
random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like_1¹
random_flip/stack_1Pack0random_flip/stateful_uniform_full_int_1:output:0!random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack_1
!random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_flip/strided_slice_1/stack
#random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_flip/strided_slice_1/stack_1
#random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_flip/strided_slice_1/stack_2Ü
random_flip/strided_slice_1StridedSlicerandom_flip/stack_1:output:0*random_flip/strided_slice_1/stack:output:0,random_flip/strided_slice_1/stack_1:output:0,random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_slice_1¿
<random_flip/stateless_random_flip_up_down/control_dependencyIdentity4random_flip/stateless_random_flip_left_right/add:z:0*
T0*C
_class9
75loc:@random_flip/stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<random_flip/stateless_random_flip_up_down/control_dependency×
/random_flip/stateless_random_flip_up_down/ShapeShapeErandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:21
/random_flip/stateless_random_flip_up_down/ShapeÈ
=random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=random_flip/stateless_random_flip_up_down/strided_slice/stackÌ
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Ì
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Þ
7random_flip/stateless_random_flip_up_down/strided_sliceStridedSlice8random_flip/stateless_random_flip_up_down/Shape:output:0Frandom_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7random_flip/stateless_random_flip_up_down/strided_slice
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2J
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip/strided_slice_1:output:0\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg* 
_output_shapes
::2a
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterÇ
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Z
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Qrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0erandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0irandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2Ú
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubOrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub÷
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMuldrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulÜ
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniform¸
9random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9random_flip/stateless_random_flip_up_down/Reshape/shape/1¸
9random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2;
9random_flip/stateless_random_flip_up_down/Reshape/shape/2¸
9random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2;
9random_flip/stateless_random_flip_up_down/Reshape/shape/3¶
7random_flip/stateless_random_flip_up_down/Reshape/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:29
7random_flip/stateless_random_flip_up_down/Reshape/shapeµ
1random_flip/stateless_random_flip_up_down/ReshapeReshapeFrandom_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0@random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1random_flip/stateless_random_flip_up_down/Reshapeá
/random_flip/stateless_random_flip_up_down/RoundRound:random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/random_flip/stateless_random_flip_up_down/Round¾
8random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/stateless_random_flip_up_down/ReverseV2/axis½
3random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0Arandom_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3random_flip/stateless_random_flip_up_down/ReverseV2
-random_flip/stateless_random_flip_up_down/mulMul3random_flip/stateless_random_flip_up_down/Round:y:0<random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-random_flip/stateless_random_flip_up_down/mul§
/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/random_flip/stateless_random_flip_up_down/sub/x
-random_flip/stateless_random_flip_up_down/subSub8random_flip/stateless_random_flip_up_down/sub/x:output:03random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-random_flip/stateless_random_flip_up_down/sub
/random_flip/stateless_random_flip_up_down/mul_1Mul1random_flip/stateless_random_flip_up_down/sub:z:0Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/random_flip/stateless_random_flip_up_down/mul_1
-random_flip/stateless_random_flip_up_down/addAddV21random_flip/stateless_random_flip_up_down/mul:z:03random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-random_flip/stateless_random_flip_up_down/add
random_rotation/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation/Shape
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2Â
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice¡
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_1/stack¥
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_1/stack_1
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2Ì
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast¡
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_2/stack¥
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_2/stack_1
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2Ì
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1®
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2&
$random_rotation/stateful_uniform/max
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation/stateful_uniform/ConstÙ
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2'
%random_rotation/stateful_uniform/Prod
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_rotation/stateful_uniform/Cast/xº
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'random_rotation/stateful_uniform/Cast_1©
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:21
/random_rotation/stateful_uniform/RngReadAndSkip¶
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4random_rotation/stateful_uniform/strided_slice/stackº
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_1º
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_2®
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask20
.random_rotation/stateful_uniform/strided_sliceÉ
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02*
(random_rotation/stateful_uniform/Bitcastº
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice_1/stack¾
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_1¾
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_2¦
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:22
0random_rotation/stateful_uniform/strided_slice_1Ï
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation/stateful_uniform/Bitcast_1À
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=random_rotation/stateful_uniform/StatelessRandomUniformV2/alg
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9random_rotation/stateful_uniform/StatelessRandomUniformV2Ò
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subï
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_rotation/stateful_uniform/mulÔ
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/y¾
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub¥
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yÄ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Ó
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/mul¥
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yÂ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2×
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_1×
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_3×
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yê
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yÂ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yÄ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ù
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yÂ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ù
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_3×
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/add×
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yð
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation/rotation_matrix/truediv_1¢
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape´
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack¸
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1¸
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2¢
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice©
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_2¿
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackÃ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2×
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_2¿
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackÃ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2×
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Neg¿
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackÃ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ù
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_3¿
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackÃ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2×
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_3¿
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackÃ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2×
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5¿
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackÃ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Û
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6¢
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Constõ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisÈ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation/rotation_matrix/concat£
random_rotation/transform/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape¨
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack¬
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1¬
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2ê
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_valueÄ
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV31random_flip/stateless_random_flip_up_down/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3®
IdentityIdentityIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip7^random_flip/stateful_uniform_full_int_1/RngReadAndSkip\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterY^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter0^random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip/stateful_uniform_full_int_1/RngReadAndSkip6random_flip/stateful_uniform_full_int_1/RngReadAndSkip2º
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2È
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterbrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2´
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgXrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2Â
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_95_layer_call_and_return_conditional_losses_164240

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ç°

H__inference_sequential_1_layer_call_and_return_conditional_losses_161829

inputsK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	
identity¢4random_flip/stateful_uniform_full_int/RngReadAndSkip¢6random_flip/stateful_uniform_full_int_1/RngReadAndSkip¢[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter¢/random_rotation/stateful_uniform/RngReadAndSkip¤
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_flip/stateful_uniform_full_int/shape¤
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+random_flip/stateful_uniform_full_int/Constí
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2,
*random_flip/stateful_uniform_full_int/Prod
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2.
,random_flip/stateful_uniform_full_int/Cast/xÉ
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int/Cast_1Â
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:26
4random_flip/stateful_uniform_full_int/RngReadAndSkipÀ
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip/stateful_uniform_full_int/strided_slice/stackÄ
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Ä
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Ì
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask25
3random_flip/stateful_uniform_full_int/strided_sliceØ
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02/
-random_flip/stateful_uniform_full_int/BitcastÄ
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice_1/stackÈ
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1È
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Ä
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:27
5random_flip/stateful_uniform_full_int/strided_slice_1Þ
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int/Bitcast_1
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_flip/stateful_uniform_full_int/algö
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2'
%random_flip/stateful_uniform_full_intz
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like±
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
random_flip/strided_slice/stack
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!random_flip/strided_slice/stack_1
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!random_flip/strided_slice/stack_2Ð
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_sliceí
?random_flip/stateless_random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?random_flip/stateless_random_flip_left_right/control_dependencyà
2random_flip/stateless_random_flip_left_right/ShapeShapeHrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:24
2random_flip/stateless_random_flip_left_right/ShapeÎ
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@random_flip/stateless_random_flip_left_right/strided_slice/stackÒ
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Ò
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2ð
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:random_flip/stateless_random_flip_left_right/strided_slice
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2M
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max®
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0* 
_output_shapes
::2d
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterÐ
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2]
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0arandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2`
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2æ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulè
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2G
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniform¾
<random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2>
<random_flip/stateless_random_flip_left_right/Reshape/shape/1¾
<random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2>
<random_flip/stateless_random_flip_left_right/Reshape/shape/2¾
<random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2>
<random_flip/stateless_random_flip_left_right/Reshape/shape/3È
:random_flip/stateless_random_flip_left_right/Reshape/shapePackCrandom_flip/stateless_random_flip_left_right/strided_slice:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Erandom_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2<
:random_flip/stateless_random_flip_left_right/Reshape/shapeÁ
4random_flip/stateless_random_flip_left_right/ReshapeReshapeIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Crandom_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4random_flip/stateless_random_flip_left_right/Reshapeê
2random_flip/stateless_random_flip_left_right/RoundRound=random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_left_right/RoundÄ
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateless_random_flip_left_right/ReverseV2/axisÉ
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0Drandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6random_flip/stateless_random_flip_left_right/ReverseV2 
0random_flip/stateless_random_flip_left_right/mulMul6random_flip/stateless_random_flip_left_right/Round:y:0?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0random_flip/stateless_random_flip_left_right/mul­
2random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2random_flip/stateless_random_flip_left_right/sub/x
0random_flip/stateless_random_flip_left_right/subSub;random_flip/stateless_random_flip_left_right/sub/x:output:06random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0random_flip/stateless_random_flip_left_right/sub«
2random_flip/stateless_random_flip_left_right/mul_1Mul4random_flip/stateless_random_flip_left_right/sub:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_left_right/mul_1
0random_flip/stateless_random_flip_left_right/addAddV24random_flip/stateless_random_flip_left_right/mul:z:06random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0random_flip/stateless_random_flip_left_right/add¨
-random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_flip/stateful_uniform_full_int_1/shape¨
-random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_flip/stateful_uniform_full_int_1/Constõ
,random_flip/stateful_uniform_full_int_1/ProdProd6random_flip/stateful_uniform_full_int_1/shape:output:06random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int_1/Prod¢
.random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip/stateful_uniform_full_int_1/Cast/xÏ
.random_flip/stateful_uniform_full_int_1/Cast_1Cast5random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.random_flip/stateful_uniform_full_int_1/Cast_1
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource7random_flip/stateful_uniform_full_int_1/Cast/x:output:02random_flip/stateful_uniform_full_int_1/Cast_1:y:05^random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:28
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipÄ
;random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;random_flip/stateful_uniform_full_int_1/strided_slice/stackÈ
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1È
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Ø
5random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask27
5random_flip/stateful_uniform_full_int_1/strided_sliceÞ
/random_flip/stateful_uniform_full_int_1/BitcastBitcast>random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int_1/BitcastÈ
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackÌ
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Ì
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Ð
7random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Frandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:29
7random_flip/stateful_uniform_full_int_1/strided_slice_1ä
1random_flip/stateful_uniform_full_int_1/Bitcast_1Bitcast@random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type023
1random_flip/stateful_uniform_full_int_1/Bitcast_1
+random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_flip/stateful_uniform_full_int_1/alg
'random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV26random_flip/stateful_uniform_full_int_1/shape:output:0:random_flip/stateful_uniform_full_int_1/Bitcast_1:output:08random_flip/stateful_uniform_full_int_1/Bitcast:output:04random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	2)
'random_flip/stateful_uniform_full_int_1~
random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like_1¹
random_flip/stack_1Pack0random_flip/stateful_uniform_full_int_1:output:0!random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack_1
!random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_flip/strided_slice_1/stack
#random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_flip/strided_slice_1/stack_1
#random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_flip/strided_slice_1/stack_2Ü
random_flip/strided_slice_1StridedSlicerandom_flip/stack_1:output:0*random_flip/strided_slice_1/stack:output:0,random_flip/strided_slice_1/stack_1:output:0,random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_slice_1¿
<random_flip/stateless_random_flip_up_down/control_dependencyIdentity4random_flip/stateless_random_flip_left_right/add:z:0*
T0*C
_class9
75loc:@random_flip/stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<random_flip/stateless_random_flip_up_down/control_dependency×
/random_flip/stateless_random_flip_up_down/ShapeShapeErandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:21
/random_flip/stateless_random_flip_up_down/ShapeÈ
=random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=random_flip/stateless_random_flip_up_down/strided_slice/stackÌ
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Ì
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Þ
7random_flip/stateless_random_flip_up_down/strided_sliceStridedSlice8random_flip/stateless_random_flip_up_down/Shape:output:0Frandom_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7random_flip/stateless_random_flip_up_down/strided_slice
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2J
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip/strided_slice_1:output:0\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg* 
_output_shapes
::2a
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterÇ
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Z
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Qrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0erandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0irandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2]
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2Ú
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubOrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub÷
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMuldrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulÜ
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2D
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniform¸
9random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2;
9random_flip/stateless_random_flip_up_down/Reshape/shape/1¸
9random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2;
9random_flip/stateless_random_flip_up_down/Reshape/shape/2¸
9random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2;
9random_flip/stateless_random_flip_up_down/Reshape/shape/3¶
7random_flip/stateless_random_flip_up_down/Reshape/shapePack@random_flip/stateless_random_flip_up_down/strided_slice:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Brandom_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:29
7random_flip/stateless_random_flip_up_down/Reshape/shapeµ
1random_flip/stateless_random_flip_up_down/ReshapeReshapeFrandom_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0@random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1random_flip/stateless_random_flip_up_down/Reshapeá
/random_flip/stateless_random_flip_up_down/RoundRound:random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/random_flip/stateless_random_flip_up_down/Round¾
8random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_flip/stateless_random_flip_up_down/ReverseV2/axis½
3random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0Arandom_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3random_flip/stateless_random_flip_up_down/ReverseV2
-random_flip/stateless_random_flip_up_down/mulMul3random_flip/stateless_random_flip_up_down/Round:y:0<random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-random_flip/stateless_random_flip_up_down/mul§
/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?21
/random_flip/stateless_random_flip_up_down/sub/x
-random_flip/stateless_random_flip_up_down/subSub8random_flip/stateless_random_flip_up_down/sub/x:output:03random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-random_flip/stateless_random_flip_up_down/sub
/random_flip/stateless_random_flip_up_down/mul_1Mul1random_flip/stateless_random_flip_up_down/sub:z:0Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ21
/random_flip/stateless_random_flip_up_down/mul_1
-random_flip/stateless_random_flip_up_down/addAddV21random_flip/stateless_random_flip_up_down/mul:z:03random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-random_flip/stateless_random_flip_up_down/add
random_rotation/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation/Shape
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2Â
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice¡
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_1/stack¥
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_1/stack_1
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2Ì
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast¡
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_2/stack¥
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_2/stack_1
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2Ì
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1®
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2&
$random_rotation/stateful_uniform/max
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation/stateful_uniform/ConstÙ
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2'
%random_rotation/stateful_uniform/Prod
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_rotation/stateful_uniform/Cast/xº
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'random_rotation/stateful_uniform/Cast_1©
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:21
/random_rotation/stateful_uniform/RngReadAndSkip¶
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4random_rotation/stateful_uniform/strided_slice/stackº
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_1º
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_2®
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask20
.random_rotation/stateful_uniform/strided_sliceÉ
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02*
(random_rotation/stateful_uniform/Bitcastº
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice_1/stack¾
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_1¾
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_2¦
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:22
0random_rotation/stateful_uniform/strided_slice_1Ï
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation/stateful_uniform/Bitcast_1À
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=random_rotation/stateful_uniform/StatelessRandomUniformV2/alg
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9random_rotation/stateful_uniform/StatelessRandomUniformV2Ò
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subï
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_rotation/stateful_uniform/mulÔ
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/y¾
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub¥
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yÄ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Ó
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/mul¥
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yÂ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2×
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_1×
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_3×
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yê
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yÂ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yÄ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ù
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yÂ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ù
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_3×
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/add×
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yð
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation/rotation_matrix/truediv_1¢
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape´
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack¸
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1¸
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2¢
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice©
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_2¿
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackÃ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2×
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_2¿
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackÃ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2×
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Neg¿
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackÃ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ù
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_3¿
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackÃ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2×
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_3¿
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackÃ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2×
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5¿
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackÃ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Û
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6¢
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Constõ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisÈ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation/rotation_matrix/concat£
random_rotation/transform/ShapeShape1random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape¨
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack¬
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1¬
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2ê
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_valueÄ
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV31random_flip/stateless_random_flip_up_down/add:z:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3®
IdentityIdentityIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityð
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip7^random_flip/stateful_uniform_full_int_1/RngReadAndSkip\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterY^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter0^random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip/stateful_uniform_full_int_1/RngReadAndSkip6random_flip/stateful_uniform_full_int_1/RngReadAndSkip2º
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2È
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterbrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2´
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgXrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2Â
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
G
+__inference_flatten_15_layer_call_fn_164265

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1614092
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ù
M
1__inference_max_pooling2d_94_layer_call_fn_164205

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_1612072
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
h
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_164060

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_94_layer_call_and_return_conditional_losses_164200

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_161355

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_164271

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_164140

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<<@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_161332

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<<@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@
 
_user_specified_nameinputs
é
Á
,stateless_random_flip_left_right_true_160967b
^stateless_random_flip_left_right_reversev2_stateless_random_flip_left_right_control_dependency-
)stateless_random_flip_left_right_identity¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis¶
*stateless_random_flip_left_right/ReverseV2	ReverseV2^stateless_random_flip_left_right_reversev2_stateless_random_flip_left_right_control_dependency8stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stateless_random_flip_left_right/ReverseV2Î
)stateless_random_flip_left_right/IdentityIdentity3stateless_random_flip_left_right/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateless_random_flip_left_right/Identity"_
)stateless_random_flip_left_right_identity2stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ì
þ
E__inference_conv2d_93_layer_call_and_return_conditional_losses_164160

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_92_layer_call_and_return_conditional_losses_164120

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ>>@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_93_layer_call_and_return_conditional_losses_161345

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

µ
*stateless_random_flip_up_down_false_161044[
Wstateless_random_flip_up_down_identity_stateless_random_flip_up_down_control_dependency*
&stateless_random_flip_up_down_identityì
&stateless_random_flip_up_down/IdentityIdentityWstateless_random_flip_up_down_identity_stateless_random_flip_up_down_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_up_down/Identity"Y
&stateless_random_flip_up_down_identity/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
£
ñ
9random_flip_stateless_random_flip_left_right_false_163305y
urandom_flip_stateless_random_flip_left_right_identity_random_flip_stateless_random_flip_left_right_control_dependency9
5random_flip_stateless_random_flip_left_right_identity¨
5random_flip/stateless_random_flip_left_right/IdentityIdentityurandom_flip_stateless_random_flip_left_right_identity_random_flip_stateless_random_flip_left_right_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/stateless_random_flip_left_right/Identity"w
5random_flip_stateless_random_flip_left_right_identity>random_flip/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ì
M
1__inference_max_pooling2d_95_layer_call_fn_164250

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_1614012
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_161185

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_random_flip_layer_call_and_return_conditional_losses_164354

inputs
identity_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_164215

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª	
d
H__inference_sequential_1_layer_call_and_return_conditional_losses_160754

inputs
identityâ
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_random_flip_layer_call_and_return_conditional_losses_1607382
random_flip/PartitionedCall
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_1607492!
random_rotation/PartitionedCall
IdentityIdentity(random_rotation/PartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

ñ
8random_flip_stateless_random_flip_left_right_true_163304z
vrandom_flip_stateless_random_flip_left_right_reversev2_random_flip_stateless_random_flip_left_right_control_dependency9
5random_flip_stateless_random_flip_left_right_identityÄ
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateless_random_flip_left_right/ReverseV2/axisò
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2vrandom_flip_stateless_random_flip_left_right_reversev2_random_flip_stateless_random_flip_left_right_control_dependencyDrandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6random_flip/stateless_random_flip_left_right/ReverseV2ò
5random_flip/stateless_random_flip_left_right/IdentityIdentity?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/stateless_random_flip_left_right/Identity"w
5random_flip_stateless_random_flip_left_right_identity>random_flip/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¹
µ
)stateless_random_flip_up_down_true_161043\
Xstateless_random_flip_up_down_reversev2_stateless_random_flip_up_down_control_dependency*
&stateless_random_flip_up_down_identity¦
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2.
,stateless_random_flip_up_down/ReverseV2/axis§
'stateless_random_flip_up_down/ReverseV2	ReverseV2Xstateless_random_flip_up_down_reversev2_stateless_random_flip_up_down_control_dependency5stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stateless_random_flip_up_down/ReverseV2Å
&stateless_random_flip_up_down/IdentityIdentity0stateless_random_flip_up_down/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_up_down/Identity"Y
&stateless_random_flip_up_down_identity/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ


*__inference_conv2d_92_layer_call_fn_164109

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1613222
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ>>@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
ì
M
1__inference_max_pooling2d_93_layer_call_fn_164170

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_1613552
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_164135

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_161378

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
µ

b
F__inference_sequential_layer_call_and_return_conditional_losses_161853

inputs
identitym
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulinputsrescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/add}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeÜ
resizing/resize/ResizeBilinearResizeBilinearrescaling_1/add:z:0resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2 
resizing/resize/ResizeBilinear
IdentityIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

ñ
8random_flip_stateless_random_flip_left_right_true_163585z
vrandom_flip_stateless_random_flip_left_right_reversev2_random_flip_stateless_random_flip_left_right_control_dependency9
5random_flip_stateless_random_flip_left_right_identityÄ
;random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateless_random_flip_left_right/ReverseV2/axisò
6random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2vrandom_flip_stateless_random_flip_left_right_reversev2_random_flip_stateless_random_flip_left_right_control_dependencyDrandom_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6random_flip/stateless_random_flip_left_right/ReverseV2ò
5random_flip/stateless_random_flip_left_right/IdentityIdentity?random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/stateless_random_flip_left_right/Identity"w
5random_flip_stateless_random_flip_left_right_identity>random_flip/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

Ë
$__inference_signature_wrapper_162187
sequential_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@

unknown_11:	@

unknown_12:@

unknown_13:@

unknown_14:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1606442
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namesequential_input
è
b
F__inference_flatten_15_layer_call_and_return_conditional_losses_161409

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
é
Á
,stateless_random_flip_left_right_true_164422b
^stateless_random_flip_left_right_reversev2_stateless_random_flip_left_right_control_dependency-
)stateless_random_flip_left_right_identity¬
/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:21
/stateless_random_flip_left_right/ReverseV2/axis¶
*stateless_random_flip_left_right/ReverseV2	ReverseV2^stateless_random_flip_left_right_reversev2_stateless_random_flip_left_right_control_dependency8stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*stateless_random_flip_left_right/ReverseV2Î
)stateless_random_flip_left_right/IdentityIdentity3stateless_random_flip_left_right/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateless_random_flip_left_right/Identity"_
)stateless_random_flip_left_right_identity2stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ð
M
1__inference_max_pooling2d_90_layer_call_fn_164050

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_1612862
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 
 
_user_specified_nameinputs
Ù
M
1__inference_max_pooling2d_90_layer_call_fn_164045

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_1611192
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

b
F__inference_sequential_layer_call_and_return_conditional_losses_163187

inputs
identitym
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulinputsrescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rescaling_1/add}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeÜ
resizing/resize/ResizeBilinearResizeBilinearrescaling_1/add:z:0resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2 
resizing/resize/ResizeBilinear
IdentityIdentity/resizing/resize/ResizeBilinear:resized_images:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
ï
G__inference_random_flip_layer_call_and_return_conditional_losses_161056

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢*stateful_uniform_full_int_1/RngReadAndSkip¢Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assert¢Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert¢Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert¢Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_like
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
&stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¿
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¤
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask20
.stateless_random_flip_left_right/strided_slice²
6stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 28
6stateless_random_flip_left_right/assert_positive/Const­
Astateless_random_flip_left_right/assert_positive/assert_less/LessLess?stateless_random_flip_left_right/assert_positive/Const:output:07stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2C
Astateless_random_flip_left_right/assert_positive/assert_less/LessÒ
Bstateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bstateless_random_flip_left_right/assert_positive/assert_less/Const·
@stateless_random_flip_left_right/assert_positive/assert_less/AllAllEstateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Kstateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2B
@stateless_random_flip_left_right/assert_positive/assert_less/All
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2K
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/Const
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2S
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0ë
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertIstateless_random_flip_left_right/assert_positive/assert_less/All:output:0Zstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2L
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert
%stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2'
%stateless_random_flip_left_right/Rank´
7stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stateless_random_flip_left_right/assert_greater_equal/y«
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.stateless_random_flip_left_right/Rank:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2D
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualº
:stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2<
:stateless_random_flip_left_right/assert_greater_equal/RankÈ
Astateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astateless_random_flip_left_right/assert_greater_equal/range/startÈ
Astateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2C
Astateless_random_flip_left_right/assert_greater_equal/range/deltaú
;stateless_random_flip_left_right/assert_greater_equal/rangeRangeJstateless_random_flip_left_right/assert_greater_equal/range/start:output:0Cstateless_random_flip_left_right/assert_greater_equal/Rank:output:0Jstateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2=
;stateless_random_flip_left_right/assert_greater_equal/range£
9stateless_random_flip_left_right/assert_greater_equal/AllAllFstateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dstateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2;
9stateless_random_flip_left_right/assert_greater_equal/Allô
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2D
Bstateless_random_flip_left_right/assert_greater_equal/Assert/Constø
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1û
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertBstateless_random_flip_left_right/assert_greater_equal/All:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.stateless_random_flip_left_right/Rank:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0K^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2E
Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assertã
3stateless_random_flip_left_right/control_dependencyIdentityinputsD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3stateless_random_flip_left_right/control_dependencyÅ
?stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxÐ
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0D^stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg½
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÆ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/mul«
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2;
9stateless_random_flip_left_right/stateless_random_uniform
'stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'stateless_random_flip_left_right/Less/yè
%stateless_random_flip_left_right/LessLess=stateless_random_flip_left_right/stateless_random_uniform:z:00stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: 2'
%stateless_random_flip_left_right/Less
 stateless_random_flip_left_rightStatelessIf)stateless_random_flip_left_right/Less:z:0<stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *@
else_branch1R/
-stateless_random_flip_left_right_false_160968*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*?
then_branch0R.
,stateless_random_flip_left_right_true_1609672"
 stateless_random_flip_left_rightÄ
)stateless_random_flip_left_right/IdentityIdentity)stateless_random_flip_left_right:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateless_random_flip_left_right/Identity
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2#
!stateful_uniform_full_int_1/shape
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!stateful_uniform_full_int_1/ConstÅ
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2"
 stateful_uniform_full_int_1/Prod
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2$
"stateful_uniform_full_int_1/Cast/x«
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"stateful_uniform_full_int_1/Cast_1¹
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:2,
*stateful_uniform_full_int_1/RngReadAndSkip¬
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/stateful_uniform_full_int_1/strided_slice/stack°
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int_1/strided_slice/stack_1°
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int_1/strided_slice/stack_2
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2+
)stateful_uniform_full_int_1/strided_sliceº
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int_1/Bitcast°
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int_1/strided_slice_1/stack´
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3stateful_uniform_full_int_1/strided_slice_1/stack_1´
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3stateful_uniform_full_int_1/strided_slice_1/stack_2
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2-
+stateful_uniform_full_int_1/strided_slice_1À
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02'
%stateful_uniform_full_int_1/Bitcast_1
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2!
stateful_uniform_full_int_1/algº
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_int_1f
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2
zeros_like_1
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2	
stack_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1¬
#stateless_random_flip_up_down/ShapeShape2stateless_random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:2%
#stateless_random_flip_up_down/Shape¹
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ23
1stateless_random_flip_up_down/strided_slice/stack´
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3stateless_random_flip_up_down/strided_slice/stack_1´
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3stateless_random_flip_up_down/strided_slice/stack_2
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2-
+stateless_random_flip_up_down/strided_slice¬
3stateless_random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 25
3stateless_random_flip_up_down/assert_positive/Const¡
>stateless_random_flip_up_down/assert_positive/assert_less/LessLess<stateless_random_flip_up_down/assert_positive/Const:output:04stateless_random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:2@
>stateless_random_flip_up_down/assert_positive/assert_less/LessÌ
?stateless_random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?stateless_random_flip_up_down/assert_positive/assert_less/Const«
=stateless_random_flip_up_down/assert_positive/assert_less/AllAllBstateless_random_flip_up_down/assert_positive/assert_less/Less:z:0Hstateless_random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2?
=stateless_random_flip_up_down/assert_positive/assert_less/Allû
Fstateless_random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2H
Fstateless_random_flip_up_down/assert_positive/assert_less/Assert/Const
Nstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2P
Nstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0±
Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertFstateless_random_flip_up_down/assert_positive/assert_less/All:output:0Wstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0P^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg*

T
2*
_output_shapes
 2I
Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert
"stateless_random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :2$
"stateless_random_flip_up_down/Rank®
4stateless_random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :26
4stateless_random_flip_up_down/assert_greater_equal/y
?stateless_random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual+stateless_random_flip_up_down/Rank:output:0=stateless_random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2A
?stateless_random_flip_up_down/assert_greater_equal/GreaterEqual´
7stateless_random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 29
7stateless_random_flip_up_down/assert_greater_equal/RankÂ
>stateless_random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2@
>stateless_random_flip_up_down/assert_greater_equal/range/startÂ
>stateless_random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2@
>stateless_random_flip_up_down/assert_greater_equal/range/deltaë
8stateless_random_flip_up_down/assert_greater_equal/rangeRangeGstateless_random_flip_up_down/assert_greater_equal/range/start:output:0@stateless_random_flip_up_down/assert_greater_equal/Rank:output:0Gstateless_random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2:
8stateless_random_flip_up_down/assert_greater_equal/range
6stateless_random_flip_up_down/assert_greater_equal/AllAllCstateless_random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Astateless_random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: 28
6stateless_random_flip_up_down/assert_greater_equal/Allî
?stateless_random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2A
?stateless_random_flip_up_down/assert_greater_equal/Assert/Constò
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2C
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_1ò
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (stateless_random_flip_up_down/Rank:0) = 2C
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_2
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (stateless_random_flip_up_down/assert_greater_equal/y:0) = 2C
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_3þ
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0þ
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1þ
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (stateless_random_flip_up_down/Rank:0) = 2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (stateless_random_flip_up_down/assert_greater_equal/y:0) = 2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4ô
@stateless_random_flip_up_down/assert_greater_equal/Assert/AssertAssert?stateless_random_flip_up_down/assert_greater_equal/All:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:0+stateless_random_flip_up_down/Rank:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0=stateless_random_flip_up_down/assert_greater_equal/y:output:0H^stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2B
@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert¦
0stateless_random_flip_up_down/control_dependencyIdentity2stateless_random_flip_left_right/Identity:output:0A^stateless_random_flip_up_down/assert_greater_equal/Assert/AssertH^stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*<
_class2
0.loc:@stateless_random_flip_left_right/Identity*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0stateless_random_flip_up_down/control_dependency¿
<stateless_random_flip_up_down/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2>
<stateless_random_flip_up_down/stateless_random_uniform/shape½
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:stateless_random_flip_up_down/stateless_random_uniform/min½
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:stateless_random_flip_up_down/stateless_random_uniform/maxÉ
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0A^stateless_random_flip_up_down/assert_greater_equal/Assert/Assert* 
_output_shapes
::2U
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter£
Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgT^stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2N
Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg«
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Rstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2Q
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2ª
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2<
:stateless_random_flip_up_down/stateless_random_uniform/subº
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2<
:stateless_random_flip_up_down/stateless_random_uniform/mul
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 28
6stateless_random_flip_up_down/stateless_random_uniform
$stateless_random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$stateless_random_flip_up_down/Less/yÜ
"stateless_random_flip_up_down/LessLess:stateless_random_flip_up_down/stateless_random_uniform:z:0-stateless_random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: 2$
"stateless_random_flip_up_down/Lessõ
stateless_random_flip_up_downStatelessIf&stateless_random_flip_up_down/Less:z:09stateless_random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *=
else_branch.R,
*stateless_random_flip_up_down_false_161044*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
then_branch-R+
)stateless_random_flip_up_down_true_1610432
stateless_random_flip_up_down»
&stateless_random_flip_up_down/IdentityIdentity&stateless_random_flip_up_down:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_up_down/Identity
IdentityIdentity/stateless_random_flip_up_down/Identity:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkipD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterA^stateless_random_flip_up_down/assert_greater_equal/Assert/AssertH^stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertM^stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgT^stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip2
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertCstateless_random_flip_left_right/assert_greater_equal/Assert/Assert2
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertJstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert2
Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertGstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert2
Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgLstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2ª
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterSstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
þ
E__inference_conv2d_90_layer_call_and_return_conditional_losses_161276

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
|
,__inference_random_flip_layer_call_fn_164350

inputs
unknown:	
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_random_flip_layer_call_and_return_conditional_losses_1610562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

m
F__inference_sequential_layer_call_and_return_conditional_losses_163153
rescaling_1_input
identitym
CastCastrescaling_1_input*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
Castm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulCast:y:0rescaling_1/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/add
resizing/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
resizing/resize/ExpandDims/dim¿
resizing/resize/ExpandDims
ExpandDimsrescaling_1/add:z:0'resizing/resize/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
resizing/resize/ExpandDims}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeã
resizing/resize/ResizeBilinearResizeBilinear#resizing/resize/ExpandDims:output:0resizing/resize/size:output:0*
T0*(
_output_shapes
:*
half_pixel_centers(2 
resizing/resize/ResizeBilinear´
resizing/resize/SqueezeSqueeze/resizing/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 2
resizing/resize/Squeezeq
IdentityIdentity resizing/resize/Squeeze:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
+
_user_specified_namerescaling_1_input
»
o
H__inference_sequential_1_layer_call_and_return_conditional_losses_163517
random_flip_input
identity
random_rotation/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
random_rotation/ExpandDims/dim½
random_rotation/ExpandDims
ExpandDimsrandom_flip_input'random_rotation/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_rotation/ExpandDims°
random_rotation/SqueezeSqueeze#random_rotation/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2
random_rotation/Squeezey
IdentityIdentity random_rotation/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
Ù
M
1__inference_max_pooling2d_92_layer_call_fn_164125

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_1611632
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_161141

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²Þ

I__inference_sequential_17_layer_call_and_return_conditional_losses_163081
sequential_inputX
Jsequential_1_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	S
Esequential_1_random_rotation_stateful_uniform_rngreadandskip_resource:	B
(conv2d_90_conv2d_readvariableop_resource: 7
)conv2d_90_biasadd_readvariableop_resource: B
(conv2d_91_conv2d_readvariableop_resource: @7
)conv2d_91_biasadd_readvariableop_resource:@B
(conv2d_92_conv2d_readvariableop_resource:@@7
)conv2d_92_biasadd_readvariableop_resource:@B
(conv2d_93_conv2d_readvariableop_resource:@@7
)conv2d_93_biasadd_readvariableop_resource:@B
(conv2d_94_conv2d_readvariableop_resource:@@7
)conv2d_94_biasadd_readvariableop_resource:@B
(conv2d_95_conv2d_readvariableop_resource:@@7
)conv2d_95_biasadd_readvariableop_resource:@:
'dense_30_matmul_readvariableop_resource:	@6
(dense_30_biasadd_readvariableop_resource:@9
'dense_31_matmul_readvariableop_resource:@6
(dense_31_biasadd_readvariableop_resource:
identity¢ conv2d_90/BiasAdd/ReadVariableOp¢conv2d_90/Conv2D/ReadVariableOp¢ conv2d_91/BiasAdd/ReadVariableOp¢conv2d_91/Conv2D/ReadVariableOp¢ conv2d_92/BiasAdd/ReadVariableOp¢conv2d_92/Conv2D/ReadVariableOp¢ conv2d_93/BiasAdd/ReadVariableOp¢conv2d_93/Conv2D/ReadVariableOp¢ conv2d_94/BiasAdd/ReadVariableOp¢conv2d_94/Conv2D/ReadVariableOp¢ conv2d_95/BiasAdd/ReadVariableOp¢conv2d_95/Conv2D/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip¢Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip¢hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter¢<sequential_1/random_rotation/stateful_uniform/RngReadAndSkip
sequential/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
sequential/rescaling_1/Cast/x
sequential/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/rescaling_1/Cast_1/xµ
sequential/rescaling_1/mulMulsequential_input&sequential/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/mulÇ
sequential/rescaling_1/addAddV2sequential/rescaling_1/mul:z:0(sequential/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/add
sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2!
sequential/resizing/resize/size
)sequential/resizing/resize/ResizeBilinearResizeBilinearsequential/rescaling_1/add:z:0(sequential/resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2+
)sequential/resizing/resize/ResizeBilinear¾
8sequential_1/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_1/random_flip/stateful_uniform_full_int/shape¾
8sequential_1/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_1/random_flip/stateful_uniform_full_int/Const¡
7sequential_1/random_flip/stateful_uniform_full_int/ProdProdAsequential_1/random_flip/stateful_uniform_full_int/shape:output:0Asequential_1/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 29
7sequential_1/random_flip/stateful_uniform_full_int/Prod¸
9sequential_1/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_1/random_flip/stateful_uniform_full_int/Cast/xð
9sequential_1/random_flip/stateful_uniform_full_int/Cast_1Cast@sequential_1/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9sequential_1/random_flip/stateful_uniform_full_int/Cast_1
Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipJsequential_1_random_flip_stateful_uniform_full_int_rngreadandskip_resourceBsequential_1/random_flip/stateful_uniform_full_int/Cast/x:output:0=sequential_1/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2C
Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipÚ
Fsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stackÞ
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_1Þ
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_2
@sequential_1/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceIsequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Osequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0Qsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Qsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2B
@sequential_1/random_flip/stateful_uniform_full_int/strided_sliceÿ
:sequential_1/random_flip/stateful_uniform_full_int/BitcastBitcastIsequential_1/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02<
:sequential_1/random_flip/stateful_uniform_full_int/BitcastÞ
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stackâ
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1â
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2
Bsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceIsequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Qsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Ssequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Ssequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2D
Bsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1
<sequential_1/random_flip/stateful_uniform_full_int/Bitcast_1BitcastKsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02>
<sequential_1/random_flip/stateful_uniform_full_int/Bitcast_1²
6sequential_1/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_1/random_flip/stateful_uniform_full_int/algÄ
2sequential_1/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2Asequential_1/random_flip/stateful_uniform_full_int/shape:output:0Esequential_1/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Csequential_1/random_flip/stateful_uniform_full_int/Bitcast:output:0?sequential_1/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	24
2sequential_1/random_flip/stateful_uniform_full_int
#sequential_1/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2%
#sequential_1/random_flip/zeros_likeå
sequential_1/random_flip/stackPack;sequential_1/random_flip/stateful_uniform_full_int:output:0,sequential_1/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:2 
sequential_1/random_flip/stack­
,sequential_1/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,sequential_1/random_flip/strided_slice/stack±
.sequential_1/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/random_flip/strided_slice/stack_1±
.sequential_1/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.sequential_1/random_flip/strided_slice/stack_2
&sequential_1/random_flip/strided_sliceStridedSlice'sequential_1/random_flip/stack:output:05sequential_1/random_flip/strided_slice/stack:output:07sequential_1/random_flip/strided_slice/stack_1:output:07sequential_1/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2(
&sequential_1/random_flip/strided_sliceÞ
Lsequential_1/random_flip/stateless_random_flip_left_right/control_dependencyIdentity:sequential/resizing/resize/ResizeBilinear:resized_images:0*
T0*<
_class2
0.loc:@sequential/resizing/resize/ResizeBilinear*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
Lsequential_1/random_flip/stateless_random_flip_left_right/control_dependency
?sequential_1/random_flip/stateless_random_flip_left_right/ShapeShapeUsequential_1/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2A
?sequential_1/random_flip/stateless_random_flip_left_right/Shapeè
Msequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Msequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stackì
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_1ì
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_2¾
Gsequential_1/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceHsequential_1/random_flip/stateless_random_flip_left_right/Shape:output:0Vsequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Xsequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Xsequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gsequential_1/random_flip/stateless_random_flip_left_right/strided_slice¼
Xsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackPsequential_1/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2Z
Xsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeõ
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minõ
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxÕ
osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter/sequential_1/random_flip/strided_slice:output:0* 
_output_shapes
::2q
osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter÷
hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgp^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2j
hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgà
ksequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2asequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0usequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ysequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0nsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
ksequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSub_sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0_sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub·
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMultsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Zsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul
Rsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Zsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0_sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniformØ
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/1Ø
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/2Ø
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/3
Gsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shapePackPsequential_1/random_flip/stateless_random_flip_left_right/strided_slice:output:0Rsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Rsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Rsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2I
Gsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shapeõ
Asequential_1/random_flip/stateless_random_flip_left_right/ReshapeReshapeVsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Psequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
Asequential_1/random_flip/stateless_random_flip_left_right/Reshape
?sequential_1/random_flip/stateless_random_flip_left_right/RoundRoundJsequential_1/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?sequential_1/random_flip/stateless_random_flip_left_right/RoundÞ
Hsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2/axisý
Csequential_1/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Usequential_1/random_flip/stateless_random_flip_left_right/control_dependency:output:0Qsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
Csequential_1/random_flip/stateless_random_flip_left_right/ReverseV2Ô
=sequential_1/random_flip/stateless_random_flip_left_right/mulMulCsequential_1/random_flip/stateless_random_flip_left_right/Round:y:0Lsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=sequential_1/random_flip/stateless_random_flip_left_right/mulÇ
?sequential_1/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?sequential_1/random_flip/stateless_random_flip_left_right/sub/xÎ
=sequential_1/random_flip/stateless_random_flip_left_right/subSubHsequential_1/random_flip/stateless_random_flip_left_right/sub/x:output:0Csequential_1/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=sequential_1/random_flip/stateless_random_flip_left_right/subß
?sequential_1/random_flip/stateless_random_flip_left_right/mul_1MulAsequential_1/random_flip/stateless_random_flip_left_right/sub:z:0Usequential_1/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?sequential_1/random_flip/stateless_random_flip_left_right/mul_1Ë
=sequential_1/random_flip/stateless_random_flip_left_right/addAddV2Asequential_1/random_flip/stateless_random_flip_left_right/mul:z:0Csequential_1/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=sequential_1/random_flip/stateless_random_flip_left_right/addÂ
:sequential_1/random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/random_flip/stateful_uniform_full_int_1/shapeÂ
:sequential_1/random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential_1/random_flip/stateful_uniform_full_int_1/Const©
9sequential_1/random_flip/stateful_uniform_full_int_1/ProdProdCsequential_1/random_flip/stateful_uniform_full_int_1/shape:output:0Csequential_1/random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2;
9sequential_1/random_flip/stateful_uniform_full_int_1/Prod¼
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast/xö
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast_1CastBsequential_1/random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast_1Ï
Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkipJsequential_1_random_flip_stateful_uniform_full_int_rngreadandskip_resourceDsequential_1/random_flip/stateful_uniform_full_int_1/Cast/x:output:0?sequential_1/random_flip/stateful_uniform_full_int_1/Cast_1:y:0B^sequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:2E
Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipÞ
Hsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stackâ
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1â
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2¦
Bsequential_1/random_flip/stateful_uniform_full_int_1/strided_sliceStridedSliceKsequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Qsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Ssequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Ssequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2D
Bsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice
<sequential_1/random_flip/stateful_uniform_full_int_1/BitcastBitcastKsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02>
<sequential_1/random_flip/stateful_uniform_full_int_1/Bitcastâ
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stackæ
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1æ
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2
Dsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSliceKsequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Ssequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Usequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Usequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2F
Dsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1
>sequential_1/random_flip/stateful_uniform_full_int_1/Bitcast_1BitcastMsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02@
>sequential_1/random_flip/stateful_uniform_full_int_1/Bitcast_1¶
8sequential_1/random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_1/random_flip/stateful_uniform_full_int_1/algÐ
4sequential_1/random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV2Csequential_1/random_flip/stateful_uniform_full_int_1/shape:output:0Gsequential_1/random_flip/stateful_uniform_full_int_1/Bitcast_1:output:0Esequential_1/random_flip/stateful_uniform_full_int_1/Bitcast:output:0Asequential_1/random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	26
4sequential_1/random_flip/stateful_uniform_full_int_1
%sequential_1/random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2'
%sequential_1/random_flip/zeros_like_1í
 sequential_1/random_flip/stack_1Pack=sequential_1/random_flip/stateful_uniform_full_int_1:output:0.sequential_1/random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2"
 sequential_1/random_flip/stack_1±
.sequential_1/random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.sequential_1/random_flip/strided_slice_1/stackµ
0sequential_1/random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0sequential_1/random_flip/strided_slice_1/stack_1µ
0sequential_1/random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0sequential_1/random_flip/strided_slice_1/stack_2ª
(sequential_1/random_flip/strided_slice_1StridedSlice)sequential_1/random_flip/stack_1:output:07sequential_1/random_flip/strided_slice_1/stack:output:09sequential_1/random_flip/strided_slice_1/stack_1:output:09sequential_1/random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2*
(sequential_1/random_flip/strided_slice_1ó
Isequential_1/random_flip/stateless_random_flip_up_down/control_dependencyIdentityAsequential_1/random_flip/stateless_random_flip_left_right/add:z:0*
T0*P
_classF
DBloc:@sequential_1/random_flip/stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Isequential_1/random_flip/stateless_random_flip_up_down/control_dependencyþ
<sequential_1/random_flip/stateless_random_flip_up_down/ShapeShapeRsequential_1/random_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2>
<sequential_1/random_flip/stateless_random_flip_up_down/Shapeâ
Jsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stackæ
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_1æ
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_2¬
Dsequential_1/random_flip/stateless_random_flip_up_down/strided_sliceStridedSliceEsequential_1/random_flip/stateless_random_flip_up_down/Shape:output:0Ssequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Usequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Usequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dsequential_1/random_flip/stateless_random_flip_up_down/strided_slice³
Usequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePackMsequential_1/random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2W
Usequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeï
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/minï
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/max¼
lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter1sequential_1/random_flip/strided_slice_1:output:0i^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg* 
_output_shapes
::2n
lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterî
esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgm^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2g
esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgÎ
hsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0rsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0vsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0ksequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
hsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/subSub\sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0\sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/sub«
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMulqsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Wsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mul
Osequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Wsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0\sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
Osequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniformÒ
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/1Ò
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/2Ò
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/3
Dsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shapePackMsequential_1/random_flip/stateless_random_flip_up_down/strided_slice:output:0Osequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Osequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Osequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shapeé
>sequential_1/random_flip/stateless_random_flip_up_down/ReshapeReshapeSsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0Msequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>sequential_1/random_flip/stateless_random_flip_up_down/Reshape
<sequential_1/random_flip/stateless_random_flip_up_down/RoundRoundGsequential_1/random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<sequential_1/random_flip/stateless_random_flip_up_down/RoundØ
Esequential_1/random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_1/random_flip/stateless_random_flip_up_down/ReverseV2/axisñ
@sequential_1/random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Rsequential_1/random_flip/stateless_random_flip_up_down/control_dependency:output:0Nsequential_1/random_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@sequential_1/random_flip/stateless_random_flip_up_down/ReverseV2È
:sequential_1/random_flip/stateless_random_flip_up_down/mulMul@sequential_1/random_flip/stateless_random_flip_up_down/Round:y:0Isequential_1/random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_1/random_flip/stateless_random_flip_up_down/mulÁ
<sequential_1/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2>
<sequential_1/random_flip/stateless_random_flip_up_down/sub/xÂ
:sequential_1/random_flip/stateless_random_flip_up_down/subSubEsequential_1/random_flip/stateless_random_flip_up_down/sub/x:output:0@sequential_1/random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_1/random_flip/stateless_random_flip_up_down/subÓ
<sequential_1/random_flip/stateless_random_flip_up_down/mul_1Mul>sequential_1/random_flip/stateless_random_flip_up_down/sub:z:0Rsequential_1/random_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<sequential_1/random_flip/stateless_random_flip_up_down/mul_1¿
:sequential_1/random_flip/stateless_random_flip_up_down/addAddV2>sequential_1/random_flip/stateless_random_flip_up_down/mul:z:0@sequential_1/random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_1/random_flip/stateless_random_flip_up_down/add¶
"sequential_1/random_rotation/ShapeShape>sequential_1/random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2$
"sequential_1/random_rotation/Shape®
0sequential_1/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_1/random_rotation/strided_slice/stack²
2sequential_1/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_rotation/strided_slice/stack_1²
2sequential_1/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_rotation/strided_slice/stack_2
*sequential_1/random_rotation/strided_sliceStridedSlice+sequential_1/random_rotation/Shape:output:09sequential_1/random_rotation/strided_slice/stack:output:0;sequential_1/random_rotation/strided_slice/stack_1:output:0;sequential_1/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_1/random_rotation/strided_slice»
2sequential_1/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/strided_slice_1/stack¿
4sequential_1/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ26
4sequential_1/random_rotation/strided_slice_1/stack_1¶
4sequential_1/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation/strided_slice_1/stack_2
,sequential_1/random_rotation/strided_slice_1StridedSlice+sequential_1/random_rotation/Shape:output:0;sequential_1/random_rotation/strided_slice_1/stack:output:0=sequential_1/random_rotation/strided_slice_1/stack_1:output:0=sequential_1/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_1/random_rotation/strided_slice_1µ
!sequential_1/random_rotation/CastCast5sequential_1/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!sequential_1/random_rotation/Cast»
2sequential_1/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/strided_slice_2/stack¿
4sequential_1/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ26
4sequential_1/random_rotation/strided_slice_2/stack_1¶
4sequential_1/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation/strided_slice_2/stack_2
,sequential_1/random_rotation/strided_slice_2StridedSlice+sequential_1/random_rotation/Shape:output:0;sequential_1/random_rotation/strided_slice_2/stack:output:0=sequential_1/random_rotation/strided_slice_2/stack_1:output:0=sequential_1/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_1/random_rotation/strided_slice_2¹
#sequential_1/random_rotation/Cast_1Cast5sequential_1/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_1/random_rotation/Cast_1Õ
3sequential_1/random_rotation/stateful_uniform/shapePack3sequential_1/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:25
3sequential_1/random_rotation/stateful_uniform/shape«
1sequential_1/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿23
1sequential_1/random_rotation/stateful_uniform/min«
1sequential_1/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?23
1sequential_1/random_rotation/stateful_uniform/max´
3sequential_1/random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/random_rotation/stateful_uniform/Const
2sequential_1/random_rotation/stateful_uniform/ProdProd<sequential_1/random_rotation/stateful_uniform/shape:output:0<sequential_1/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/stateful_uniform/Prod®
4sequential_1/random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_1/random_rotation/stateful_uniform/Cast/xá
4sequential_1/random_rotation/stateful_uniform/Cast_1Cast;sequential_1/random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4sequential_1/random_rotation/stateful_uniform/Cast_1ê
<sequential_1/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkipEsequential_1_random_rotation_stateful_uniform_rngreadandskip_resource=sequential_1/random_rotation/stateful_uniform/Cast/x:output:08sequential_1/random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:2>
<sequential_1/random_rotation/stateful_uniform/RngReadAndSkipÐ
Asequential_1/random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_1/random_rotation/stateful_uniform/strided_slice/stackÔ
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_1Ô
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_2ü
;sequential_1/random_rotation/stateful_uniform/strided_sliceStridedSliceDsequential_1/random_rotation/stateful_uniform/RngReadAndSkip:value:0Jsequential_1/random_rotation/stateful_uniform/strided_slice/stack:output:0Lsequential_1/random_rotation/stateful_uniform/strided_slice/stack_1:output:0Lsequential_1/random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2=
;sequential_1/random_rotation/stateful_uniform/strided_sliceð
5sequential_1/random_rotation/stateful_uniform/BitcastBitcastDsequential_1/random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type027
5sequential_1/random_rotation/stateful_uniform/BitcastÔ
Csequential_1/random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_1/random_rotation/stateful_uniform/strided_slice_1/stackØ
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_1Ø
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_2ô
=sequential_1/random_rotation/stateful_uniform/strided_slice_1StridedSliceDsequential_1/random_rotation/stateful_uniform/RngReadAndSkip:value:0Lsequential_1/random_rotation/stateful_uniform/strided_slice_1/stack:output:0Nsequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Nsequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2?
=sequential_1/random_rotation/stateful_uniform/strided_slice_1ö
7sequential_1/random_rotation/stateful_uniform/Bitcast_1BitcastFsequential_1/random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type029
7sequential_1/random_rotation/stateful_uniform/Bitcast_1Ú
Jsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2/algæ
Fsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_1/random_rotation/stateful_uniform/shape:output:0@sequential_1/random_rotation/stateful_uniform/Bitcast_1:output:0>sequential_1/random_rotation/stateful_uniform/Bitcast:output:0Ssequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
Fsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2
1sequential_1/random_rotation/stateful_uniform/subSub:sequential_1/random_rotation/stateful_uniform/max:output:0:sequential_1/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 23
1sequential_1/random_rotation/stateful_uniform/sub£
1sequential_1/random_rotation/stateful_uniform/mulMulOsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2:output:05sequential_1/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_1/random_rotation/stateful_uniform/mul
-sequential_1/random_rotation/stateful_uniformAddV25sequential_1/random_rotation/stateful_uniform/mul:z:0:sequential_1/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_1/random_rotation/stateful_uniform­
2sequential_1/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_1/random_rotation/rotation_matrix/sub/yò
0sequential_1/random_rotation/rotation_matrix/subSub'sequential_1/random_rotation/Cast_1:y:0;sequential_1/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 22
0sequential_1/random_rotation/rotation_matrix/subÌ
0sequential_1/random_rotation/rotation_matrix/CosCos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/Cos±
4sequential_1/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_1/yø
2sequential_1/random_rotation/rotation_matrix/sub_1Sub'sequential_1/random_rotation/Cast_1:y:0=sequential_1/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_1
0sequential_1/random_rotation/rotation_matrix/mulMul4sequential_1/random_rotation/rotation_matrix/Cos:y:06sequential_1/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/mulÌ
0sequential_1/random_rotation/rotation_matrix/SinSin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/Sin±
4sequential_1/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_2/yö
2sequential_1/random_rotation/rotation_matrix/sub_2Sub%sequential_1/random_rotation/Cast:y:0=sequential_1/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_2
2sequential_1/random_rotation/rotation_matrix/mul_1Mul4sequential_1/random_rotation/rotation_matrix/Sin:y:06sequential_1/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/mul_1
2sequential_1/random_rotation/rotation_matrix/sub_3Sub4sequential_1/random_rotation/rotation_matrix/mul:z:06sequential_1/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/sub_3
2sequential_1/random_rotation/rotation_matrix/sub_4Sub4sequential_1/random_rotation/rotation_matrix/sub:z:06sequential_1/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/sub_4µ
6sequential_1/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @28
6sequential_1/random_rotation/rotation_matrix/truediv/y
4sequential_1/random_rotation/rotation_matrix/truedivRealDiv6sequential_1/random_rotation/rotation_matrix/sub_4:z:0?sequential_1/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_1/random_rotation/rotation_matrix/truediv±
4sequential_1/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_5/yö
2sequential_1/random_rotation/rotation_matrix/sub_5Sub%sequential_1/random_rotation/Cast:y:0=sequential_1/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_5Ð
2sequential_1/random_rotation/rotation_matrix/Sin_1Sin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Sin_1±
4sequential_1/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_6/yø
2sequential_1/random_rotation/rotation_matrix/sub_6Sub'sequential_1/random_rotation/Cast_1:y:0=sequential_1/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_6
2sequential_1/random_rotation/rotation_matrix/mul_2Mul6sequential_1/random_rotation/rotation_matrix/Sin_1:y:06sequential_1/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/mul_2Ð
2sequential_1/random_rotation/rotation_matrix/Cos_1Cos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Cos_1±
4sequential_1/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_7/yö
2sequential_1/random_rotation/rotation_matrix/sub_7Sub%sequential_1/random_rotation/Cast:y:0=sequential_1/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_7
2sequential_1/random_rotation/rotation_matrix/mul_3Mul6sequential_1/random_rotation/rotation_matrix/Cos_1:y:06sequential_1/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/mul_3
0sequential_1/random_rotation/rotation_matrix/addAddV26sequential_1/random_rotation/rotation_matrix/mul_2:z:06sequential_1/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/add
2sequential_1/random_rotation/rotation_matrix/sub_8Sub6sequential_1/random_rotation/rotation_matrix/sub_5:z:04sequential_1/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/sub_8¹
8sequential_1/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2:
8sequential_1/random_rotation/rotation_matrix/truediv_1/y¤
6sequential_1/random_rotation/rotation_matrix/truediv_1RealDiv6sequential_1/random_rotation/rotation_matrix/sub_8:z:0Asequential_1/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_1/random_rotation/rotation_matrix/truediv_1É
2sequential_1/random_rotation/rotation_matrix/ShapeShape1sequential_1/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:24
2sequential_1/random_rotation/rotation_matrix/ShapeÎ
@sequential_1/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_1/random_rotation/rotation_matrix/strided_slice/stackÒ
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_1Ò
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_2ð
:sequential_1/random_rotation/rotation_matrix/strided_sliceStridedSlice;sequential_1/random_rotation/rotation_matrix/Shape:output:0Isequential_1/random_rotation/rotation_matrix/strided_slice/stack:output:0Ksequential_1/random_rotation/rotation_matrix/strided_slice/stack_1:output:0Ksequential_1/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_1/random_rotation/rotation_matrix/strided_sliceÐ
2sequential_1/random_rotation/rotation_matrix/Cos_2Cos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Cos_2Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_1/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_1StridedSlice6sequential_1/random_rotation/rotation_matrix/Cos_2:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_1/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_1Ð
2sequential_1/random_rotation/rotation_matrix/Sin_2Sin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Sin_2Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_2/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_2StridedSlice6sequential_1/random_rotation/rotation_matrix/Sin_2:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_2/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_2ä
0sequential_1/random_rotation/rotation_matrix/NegNegEsequential_1/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/NegÙ
Bsequential_1/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_3/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_2§
<sequential_1/random_rotation/rotation_matrix/strided_slice_3StridedSlice8sequential_1/random_rotation/rotation_matrix/truediv:z:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_3/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_3Ð
2sequential_1/random_rotation/rotation_matrix/Sin_3Sin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Sin_3Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_4/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_4StridedSlice6sequential_1/random_rotation/rotation_matrix/Sin_3:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_4/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_4Ð
2sequential_1/random_rotation/rotation_matrix/Cos_3Cos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Cos_3Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_5/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_5StridedSlice6sequential_1/random_rotation/rotation_matrix/Cos_3:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_5/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_5Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_6/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_2©
<sequential_1/random_rotation/rotation_matrix/strided_slice_6StridedSlice:sequential_1/random_rotation/rotation_matrix/truediv_1:z:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_6/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_6¼
;sequential_1/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/random_rotation/rotation_matrix/zeros/packed/1·
9sequential_1/random_rotation/rotation_matrix/zeros/packedPackCsequential_1/random_rotation/rotation_matrix/strided_slice:output:0Dsequential_1/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/random_rotation/rotation_matrix/zeros/packed¹
8sequential_1/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8sequential_1/random_rotation/rotation_matrix/zeros/Const©
2sequential_1/random_rotation/rotation_matrix/zerosFillBsequential_1/random_rotation/rotation_matrix/zeros/packed:output:0Asequential_1/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/zeros¶
8sequential_1/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_1/random_rotation/rotation_matrix/concat/axisÊ
3sequential_1/random_rotation/rotation_matrix/concatConcatV2Esequential_1/random_rotation/rotation_matrix/strided_slice_1:output:04sequential_1/random_rotation/rotation_matrix/Neg:y:0Esequential_1/random_rotation/rotation_matrix/strided_slice_3:output:0Esequential_1/random_rotation/rotation_matrix/strided_slice_4:output:0Esequential_1/random_rotation/rotation_matrix/strided_slice_5:output:0Esequential_1/random_rotation/rotation_matrix/strided_slice_6:output:0;sequential_1/random_rotation/rotation_matrix/zeros:output:0Asequential_1/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3sequential_1/random_rotation/rotation_matrix/concatÊ
,sequential_1/random_rotation/transform/ShapeShape>sequential_1/random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2.
,sequential_1/random_rotation/transform/ShapeÂ
:sequential_1/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/random_rotation/transform/strided_slice/stackÆ
<sequential_1/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_rotation/transform/strided_slice/stack_1Æ
<sequential_1/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_rotation/transform/strided_slice/stack_2¸
4sequential_1/random_rotation/transform/strided_sliceStridedSlice5sequential_1/random_rotation/transform/Shape:output:0Csequential_1/random_rotation/transform/strided_slice/stack:output:0Esequential_1/random_rotation/transform/strided_slice/stack_1:output:0Esequential_1/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:26
4sequential_1/random_rotation/transform/strided_slice«
1sequential_1/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential_1/random_rotation/transform/fill_value
Asequential_1/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3>sequential_1/random_flip/stateless_random_flip_up_down/add:z:0<sequential_1/random_rotation/rotation_matrix/concat:output:0=sequential_1/random_rotation/transform/strided_slice:output:0:sequential_1/random_rotation/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2C
Asequential_1/random_rotation/transform/ImageProjectiveTransformV3³
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_90/Conv2D/ReadVariableOp
conv2d_90/Conv2DConv2DVsequential_1/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2
conv2d_90/Conv2Dª
 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_90/BiasAdd/ReadVariableOp²
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/BiasAdd
conv2d_90/ReluReluconv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/ReluÊ
max_pooling2d_90/MaxPoolMaxPoolconv2d_90/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_90/MaxPool³
conv2d_91/Conv2D/ReadVariableOpReadVariableOp(conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_91/Conv2D/ReadVariableOpÝ
conv2d_91/Conv2DConv2D!max_pooling2d_90/MaxPool:output:0'conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2
conv2d_91/Conv2Dª
 conv2d_91/BiasAdd/ReadVariableOpReadVariableOp)conv2d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_91/BiasAdd/ReadVariableOp°
conv2d_91/BiasAddBiasAddconv2d_91/Conv2D:output:0(conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/BiasAdd~
conv2d_91/ReluReluconv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/ReluÊ
max_pooling2d_91/MaxPoolMaxPoolconv2d_91/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_91/MaxPool³
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_92/Conv2D/ReadVariableOpÝ
conv2d_92/Conv2DConv2D!max_pooling2d_91/MaxPool:output:0'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2
conv2d_92/Conv2Dª
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp°
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/ReluÊ
max_pooling2d_92/MaxPoolMaxPoolconv2d_92/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_92/MaxPool³
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_93/Conv2D/ReadVariableOpÝ
conv2d_93/Conv2DConv2D!max_pooling2d_92/MaxPool:output:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_93/Conv2Dª
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp°
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/ReluÊ
max_pooling2d_93/MaxPoolMaxPoolconv2d_93/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_93/MaxPool³
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_94/Conv2D/ReadVariableOpÝ
conv2d_94/Conv2DConv2D!max_pooling2d_93/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_94/Conv2Dª
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp°
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/ReluÊ
max_pooling2d_94/MaxPoolMaxPoolconv2d_94/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_94/MaxPool³
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_95/Conv2D/ReadVariableOpÝ
conv2d_95/Conv2DConv2D!max_pooling2d_94/MaxPool:output:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_95/Conv2Dª
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp°
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/ReluÊ
max_pooling2d_95/MaxPoolMaxPoolconv2d_95/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_95/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_15/Const¤
flatten_15/ReshapeReshape!max_pooling2d_95/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_15/Reshape©
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_30/MatMul/ReadVariableOp£
dense_30/MatMulMatMulflatten_15/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/MatMul§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp¥
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_31/MatMul/ReadVariableOp£
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMul§
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp¥
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/BiasAdd|
dense_31/SoftmaxSoftmaxdense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/Softmaxu
IdentityIdentitydense_31/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityï	
NoOpNoOp!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp!^conv2d_91/BiasAdd/ReadVariableOp ^conv2d_91/Conv2D/ReadVariableOp!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOpB^sequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipD^sequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipi^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgp^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterf^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgm^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter=^sequential_1/random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp2D
 conv2d_91/BiasAdd/ReadVariableOp conv2d_91/BiasAdd/ReadVariableOp2B
conv2d_91/Conv2D/ReadVariableOpconv2d_91/Conv2D/ReadVariableOp2D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2
Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipAsequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip2
Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipCsequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip2Ô
hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlghsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2â
osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterosequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2Î
esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgesequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2Ü
lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterlsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter2|
<sequential_1/random_rotation/stateful_uniform/RngReadAndSkip<sequential_1/random_rotation/stateful_uniform/RngReadAndSkip:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namesequential_input
å
R
+__inference_sequential_layer_call_fn_163101
rescaling_1_input
identityÌ
PartitionedCallPartitionedCallrescaling_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1607032
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
+
_user_specified_namerescaling_1_input
Ö
H
,__inference_rescaling_1_layer_call_fn_164316

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_1606602
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_161309

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@
 
_user_specified_nameinputs
ù
T
-__inference_sequential_1_layer_call_fn_163192
random_flip_input
identityÖ
PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1607542
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
¬
h
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_164095

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_164220

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_91_layer_call_and_return_conditional_losses_164080

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_92_layer_call_and_return_conditional_losses_161322

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ>>@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@
 
_user_specified_nameinputs
¢
d
H__inference_sequential_1_layer_call_and_return_conditional_losses_161263

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä
G
+__inference_sequential_layer_call_fn_163091

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1606742
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
ô

)__inference_dense_30_layer_call_fn_164280

inputs
unknown:	@
	unknown_0:@
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_1614222
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
H__inference_sequential_1_layer_call_and_return_conditional_losses_163236

inputs
identity
random_rotation/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
random_rotation/ExpandDims/dim²
random_rotation/ExpandDims
ExpandDimsinputs'random_rotation/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_rotation/ExpandDims°
random_rotation/SqueezeSqueeze#random_rotation/ExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2
random_rotation/Squeezey
IdentityIdentity random_rotation/Squeeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
h
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_161286

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿþþ :Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 
 
_user_specified_nameinputs
ï
b
F__inference_sequential_layer_call_and_return_conditional_losses_163139

inputs
identityb
CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
Castm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulCast:y:0rescaling_1/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/add
resizing/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
resizing/resize/ExpandDims/dim¿
resizing/resize/ExpandDims
ExpandDimsrescaling_1/add:z:0'resizing/resize/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
resizing/resize/ExpandDims}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeã
resizing/resize/ResizeBilinearResizeBilinear#resizing/resize/ExpandDims:output:0resizing/resize/size:output:0*
T0*(
_output_shapes
:*
half_pixel_centers(2 
resizing/resize/ResizeBilinear´
resizing/resize/SqueezeSqueeze/resizing/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 2
resizing/resize/Squeezeq
IdentityIdentity resizing/resize/Squeeze:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
è
ÿ
.__inference_sequential_17_layer_call_fn_162302

inputs
unknown:	
	unknown_0:	#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:	@

unknown_14:@

unknown_15:@

unknown_16:
identity¢StatefulPartitionedCallÎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1619522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£
ñ
9random_flip_stateless_random_flip_left_right_false_163586y
urandom_flip_stateless_random_flip_left_right_identity_random_flip_stateless_random_flip_left_right_control_dependency9
5random_flip_stateless_random_flip_left_right_identity¨
5random_flip/stateless_random_flip_left_right/IdentityIdentityurandom_flip_stateless_random_flip_left_right_identity_random_flip_stateless_random_flip_left_right_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/stateless_random_flip_left_right/Identity"w
5random_flip_stateless_random_flip_left_right_identity>random_flip/stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w

__inference__traced_save_164861
file_prefix/
+savev2_conv2d_90_kernel_read_readvariableop-
)savev2_conv2d_90_bias_read_readvariableop/
+savev2_conv2d_91_kernel_read_readvariableop-
)savev2_conv2d_91_bias_read_readvariableop/
+savev2_conv2d_92_kernel_read_readvariableop-
)savev2_conv2d_92_bias_read_readvariableop/
+savev2_conv2d_93_kernel_read_readvariableop-
)savev2_conv2d_93_bias_read_readvariableop/
+savev2_conv2d_94_kernel_read_readvariableop-
)savev2_conv2d_94_bias_read_readvariableop/
+savev2_conv2d_95_kernel_read_readvariableop-
)savev2_conv2d_95_bias_read_readvariableop.
*savev2_dense_30_kernel_read_readvariableop,
(savev2_dense_30_bias_read_readvariableop.
*savev2_dense_31_kernel_read_readvariableop,
(savev2_dense_31_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_90_kernel_m_read_readvariableop4
0savev2_adam_conv2d_90_bias_m_read_readvariableop6
2savev2_adam_conv2d_91_kernel_m_read_readvariableop4
0savev2_adam_conv2d_91_bias_m_read_readvariableop6
2savev2_adam_conv2d_92_kernel_m_read_readvariableop4
0savev2_adam_conv2d_92_bias_m_read_readvariableop6
2savev2_adam_conv2d_93_kernel_m_read_readvariableop4
0savev2_adam_conv2d_93_bias_m_read_readvariableop6
2savev2_adam_conv2d_94_kernel_m_read_readvariableop4
0savev2_adam_conv2d_94_bias_m_read_readvariableop6
2savev2_adam_conv2d_95_kernel_m_read_readvariableop4
0savev2_adam_conv2d_95_bias_m_read_readvariableop5
1savev2_adam_dense_30_kernel_m_read_readvariableop3
/savev2_adam_dense_30_bias_m_read_readvariableop5
1savev2_adam_dense_31_kernel_m_read_readvariableop3
/savev2_adam_dense_31_bias_m_read_readvariableop6
2savev2_adam_conv2d_90_kernel_v_read_readvariableop4
0savev2_adam_conv2d_90_bias_v_read_readvariableop6
2savev2_adam_conv2d_91_kernel_v_read_readvariableop4
0savev2_adam_conv2d_91_bias_v_read_readvariableop6
2savev2_adam_conv2d_92_kernel_v_read_readvariableop4
0savev2_adam_conv2d_92_bias_v_read_readvariableop6
2savev2_adam_conv2d_93_kernel_v_read_readvariableop4
0savev2_adam_conv2d_93_bias_v_read_readvariableop6
2savev2_adam_conv2d_94_kernel_v_read_readvariableop4
0savev2_adam_conv2d_94_bias_v_read_readvariableop6
2savev2_adam_conv2d_95_kernel_v_read_readvariableop4
0savev2_adam_conv2d_95_bias_v_read_readvariableop5
1savev2_adam_dense_30_kernel_v_read_readvariableop3
/savev2_adam_dense_30_bias_v_read_readvariableop5
1savev2_adam_dense_31_kernel_v_read_readvariableop3
/savev2_adam_dense_31_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

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
ShardedFilename¦!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*¸ 
value® B« <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_90_kernel_read_readvariableop)savev2_conv2d_90_bias_read_readvariableop+savev2_conv2d_91_kernel_read_readvariableop)savev2_conv2d_91_bias_read_readvariableop+savev2_conv2d_92_kernel_read_readvariableop)savev2_conv2d_92_bias_read_readvariableop+savev2_conv2d_93_kernel_read_readvariableop)savev2_conv2d_93_bias_read_readvariableop+savev2_conv2d_94_kernel_read_readvariableop)savev2_conv2d_94_bias_read_readvariableop+savev2_conv2d_95_kernel_read_readvariableop)savev2_conv2d_95_bias_read_readvariableop*savev2_dense_30_kernel_read_readvariableop(savev2_dense_30_bias_read_readvariableop*savev2_dense_31_kernel_read_readvariableop(savev2_dense_31_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_90_kernel_m_read_readvariableop0savev2_adam_conv2d_90_bias_m_read_readvariableop2savev2_adam_conv2d_91_kernel_m_read_readvariableop0savev2_adam_conv2d_91_bias_m_read_readvariableop2savev2_adam_conv2d_92_kernel_m_read_readvariableop0savev2_adam_conv2d_92_bias_m_read_readvariableop2savev2_adam_conv2d_93_kernel_m_read_readvariableop0savev2_adam_conv2d_93_bias_m_read_readvariableop2savev2_adam_conv2d_94_kernel_m_read_readvariableop0savev2_adam_conv2d_94_bias_m_read_readvariableop2savev2_adam_conv2d_95_kernel_m_read_readvariableop0savev2_adam_conv2d_95_bias_m_read_readvariableop1savev2_adam_dense_30_kernel_m_read_readvariableop/savev2_adam_dense_30_bias_m_read_readvariableop1savev2_adam_dense_31_kernel_m_read_readvariableop/savev2_adam_dense_31_bias_m_read_readvariableop2savev2_adam_conv2d_90_kernel_v_read_readvariableop0savev2_adam_conv2d_90_bias_v_read_readvariableop2savev2_adam_conv2d_91_kernel_v_read_readvariableop0savev2_adam_conv2d_91_bias_v_read_readvariableop2savev2_adam_conv2d_92_kernel_v_read_readvariableop0savev2_adam_conv2d_92_bias_v_read_readvariableop2savev2_adam_conv2d_93_kernel_v_read_readvariableop0savev2_adam_conv2d_93_bias_v_read_readvariableop2savev2_adam_conv2d_94_kernel_v_read_readvariableop0savev2_adam_conv2d_94_bias_v_read_readvariableop2savev2_adam_conv2d_95_kernel_v_read_readvariableop0savev2_adam_conv2d_95_bias_v_read_readvariableop1savev2_adam_dense_30_kernel_v_read_readvariableop/savev2_adam_dense_30_bias_v_read_readvariableop1savev2_adam_dense_31_kernel_v_read_readvariableop/savev2_adam_dense_31_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<			2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*Ê
_input_shapes¸
µ: : : : @:@:@@:@:@@:@:@@:@:@@:@:	@:@:@:: : : : : ::: : : : : : : @:@:@@:@:@@:@:@@:@:@@:@:	@:@:@:: : : @:@:@@:@:@@:@:@@:@:@@:@:	@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:, (
&
_output_shapes
:@@: !

_output_shapes
:@:,"(
&
_output_shapes
:@@: #

_output_shapes
:@:,$(
&
_output_shapes
:@@: %

_output_shapes
:@:,&(
&
_output_shapes
:@@: '

_output_shapes
:@:%(!

_output_shapes
:	@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
: @: /

_output_shapes
:@:,0(
&
_output_shapes
:@@: 1

_output_shapes
:@:,2(
&
_output_shapes
:@@: 3

_output_shapes
:@:,4(
&
_output_shapes
:@@: 5

_output_shapes
:@:,6(
&
_output_shapes
:@@: 7

_output_shapes
:@:%8!

_output_shapes
:	@: 9

_output_shapes
:@:$: 

_output_shapes

:@: ;

_output_shapes
::<

_output_shapes
: 

å
6random_flip_stateless_random_flip_up_down_false_163662s
orandom_flip_stateless_random_flip_up_down_identity_random_flip_stateless_random_flip_up_down_control_dependency6
2random_flip_stateless_random_flip_up_down_identity
2random_flip/stateless_random_flip_up_down/IdentityIdentityorandom_flip_stateless_random_flip_up_down_identity_random_flip_stateless_random_flip_up_down_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_up_down/Identity"q
2random_flip_stateless_random_flip_up_down_identity;random_flip/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ù
M
1__inference_max_pooling2d_93_layer_call_fn_164165

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_1611852
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ág
á
I__inference_sequential_17_layer_call_and_return_conditional_losses_162417

inputsB
(conv2d_90_conv2d_readvariableop_resource: 7
)conv2d_90_biasadd_readvariableop_resource: B
(conv2d_91_conv2d_readvariableop_resource: @7
)conv2d_91_biasadd_readvariableop_resource:@B
(conv2d_92_conv2d_readvariableop_resource:@@7
)conv2d_92_biasadd_readvariableop_resource:@B
(conv2d_93_conv2d_readvariableop_resource:@@7
)conv2d_93_biasadd_readvariableop_resource:@B
(conv2d_94_conv2d_readvariableop_resource:@@7
)conv2d_94_biasadd_readvariableop_resource:@B
(conv2d_95_conv2d_readvariableop_resource:@@7
)conv2d_95_biasadd_readvariableop_resource:@:
'dense_30_matmul_readvariableop_resource:	@6
(dense_30_biasadd_readvariableop_resource:@9
'dense_31_matmul_readvariableop_resource:@6
(dense_31_biasadd_readvariableop_resource:
identity¢ conv2d_90/BiasAdd/ReadVariableOp¢conv2d_90/Conv2D/ReadVariableOp¢ conv2d_91/BiasAdd/ReadVariableOp¢conv2d_91/Conv2D/ReadVariableOp¢ conv2d_92/BiasAdd/ReadVariableOp¢conv2d_92/Conv2D/ReadVariableOp¢ conv2d_93/BiasAdd/ReadVariableOp¢conv2d_93/Conv2D/ReadVariableOp¢ conv2d_94/BiasAdd/ReadVariableOp¢conv2d_94/Conv2D/ReadVariableOp¢ conv2d_95/BiasAdd/ReadVariableOp¢conv2d_95/Conv2D/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢dense_31/MatMul/ReadVariableOp
sequential/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
sequential/rescaling_1/Cast/x
sequential/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/rescaling_1/Cast_1/x«
sequential/rescaling_1/mulMulinputs&sequential/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/mulÇ
sequential/rescaling_1/addAddV2sequential/rescaling_1/mul:z:0(sequential/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/add
sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2!
sequential/resizing/resize/size
)sequential/resizing/resize/ResizeBilinearResizeBilinearsequential/rescaling_1/add:z:0(sequential/resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2+
)sequential/resizing/resize/ResizeBilinear³
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_90/Conv2D/ReadVariableOpø
conv2d_90/Conv2DConv2D:sequential/resizing/resize/ResizeBilinear:resized_images:0'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2
conv2d_90/Conv2Dª
 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_90/BiasAdd/ReadVariableOp²
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/BiasAdd
conv2d_90/ReluReluconv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/ReluÊ
max_pooling2d_90/MaxPoolMaxPoolconv2d_90/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_90/MaxPool³
conv2d_91/Conv2D/ReadVariableOpReadVariableOp(conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_91/Conv2D/ReadVariableOpÝ
conv2d_91/Conv2DConv2D!max_pooling2d_90/MaxPool:output:0'conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2
conv2d_91/Conv2Dª
 conv2d_91/BiasAdd/ReadVariableOpReadVariableOp)conv2d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_91/BiasAdd/ReadVariableOp°
conv2d_91/BiasAddBiasAddconv2d_91/Conv2D:output:0(conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/BiasAdd~
conv2d_91/ReluReluconv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/ReluÊ
max_pooling2d_91/MaxPoolMaxPoolconv2d_91/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_91/MaxPool³
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_92/Conv2D/ReadVariableOpÝ
conv2d_92/Conv2DConv2D!max_pooling2d_91/MaxPool:output:0'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2
conv2d_92/Conv2Dª
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp°
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/ReluÊ
max_pooling2d_92/MaxPoolMaxPoolconv2d_92/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_92/MaxPool³
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_93/Conv2D/ReadVariableOpÝ
conv2d_93/Conv2DConv2D!max_pooling2d_92/MaxPool:output:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_93/Conv2Dª
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp°
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/ReluÊ
max_pooling2d_93/MaxPoolMaxPoolconv2d_93/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_93/MaxPool³
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_94/Conv2D/ReadVariableOpÝ
conv2d_94/Conv2DConv2D!max_pooling2d_93/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_94/Conv2Dª
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp°
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/ReluÊ
max_pooling2d_94/MaxPoolMaxPoolconv2d_94/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_94/MaxPool³
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_95/Conv2D/ReadVariableOpÝ
conv2d_95/Conv2DConv2D!max_pooling2d_94/MaxPool:output:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_95/Conv2Dª
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp°
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/ReluÊ
max_pooling2d_95/MaxPoolMaxPoolconv2d_95/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_95/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_15/Const¤
flatten_15/ReshapeReshape!max_pooling2d_95/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_15/Reshape©
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_30/MatMul/ReadVariableOp£
dense_30/MatMulMatMulflatten_15/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/MatMul§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp¥
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_31/MatMul/ReadVariableOp£
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMul§
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp¥
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/BiasAdd|
dense_31/SoftmaxSoftmaxdense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/Softmaxu
IdentityIdentitydense_31/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp!^conv2d_91/BiasAdd/ReadVariableOp ^conv2d_91/Conv2D/ReadVariableOp!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp2D
 conv2d_91/BiasAdd/ReadVariableOp conv2d_91/BiasAdd/ReadVariableOp2B
conv2d_91/Conv2D/ReadVariableOpconv2d_91/Conv2D/ReadVariableOp2D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
H__inference_sequential_1_layer_call_and_return_conditional_losses_161078

inputs 
random_flip_161071:	$
random_rotation_161074:	
identity¢#random_flip/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCall
#random_flip/StatefulPartitionedCallStatefulPartitionedCallinputsrandom_flip_161071*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_random_flip_layer_call_and_return_conditional_losses_1610562%
#random_flip/StatefulPartitionedCallÅ
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall,random_flip/StatefulPartitionedCall:output:0random_rotation_161074*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_1608862)
'random_rotation/StatefulPartitionedCall
IdentityIdentity0random_rotation/StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp$^random_flip/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2J
#random_flip/StatefulPartitionedCall#random_flip/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


.__inference_sequential_17_layer_call_fn_162343
sequential_input
unknown:	
	unknown_0:	#
	unknown_1: 
	unknown_2: #
	unknown_3: @
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@$

unknown_11:@@

unknown_12:@

unknown_13:	@

unknown_14:@

unknown_15:@

unknown_16:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1619522
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namesequential_input
á	
b
F__inference_sequential_layer_call_and_return_conditional_losses_160703

inputs
identityb
CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
Castä
rescaling_1/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_1606602
rescaling_1/PartitionedCallï
resizing/PartitionedCallPartitionedCall$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_1606712
resizing/PartitionedCallr
IdentityIdentity!resizing/PartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
Þ
L
0__inference_random_rotation_layer_call_fn_164516

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_1607492
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ

)__inference_dense_31_layer_call_fn_164300

inputs
unknown:@
	unknown_0:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_1614392
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

å
6random_flip_stateless_random_flip_up_down_false_163381s
orandom_flip_stateless_random_flip_up_down_identity_random_flip_stateless_random_flip_up_down_control_dependency6
2random_flip_stateless_random_flip_up_down_identity
2random_flip/stateless_random_flip_up_down/IdentityIdentityorandom_flip_stateless_random_flip_up_down_identity_random_flip_stateless_random_flip_up_down_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_up_down/Identity"q
2random_flip_stateless_random_flip_up_down_identity;random_flip/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
h
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_161229

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_164100

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@
 
_user_specified_nameinputs
ù	
å
5random_flip_stateless_random_flip_up_down_true_163380t
prandom_flip_stateless_random_flip_up_down_reversev2_random_flip_stateless_random_flip_up_down_control_dependency6
2random_flip_stateless_random_flip_up_down_identity¾
8random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip/stateless_random_flip_up_down/ReverseV2/axisã
3random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2prandom_flip_stateless_random_flip_up_down_reversev2_random_flip_stateless_random_flip_up_down_control_dependencyArandom_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3random_flip/stateless_random_flip_up_down/ReverseV2é
2random_flip/stateless_random_flip_up_down/IdentityIdentity<random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_up_down/Identity"q
2random_flip_stateless_random_flip_up_down_identity;random_flip/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

g
K__inference_random_rotation_layer_call_and_return_conditional_losses_164530

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDims
SqueezeSqueezeExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ

I__inference_sequential_17_layer_call_and_return_conditional_losses_162712

inputsX
Jsequential_1_random_flip_stateful_uniform_full_int_rngreadandskip_resource:	S
Esequential_1_random_rotation_stateful_uniform_rngreadandskip_resource:	B
(conv2d_90_conv2d_readvariableop_resource: 7
)conv2d_90_biasadd_readvariableop_resource: B
(conv2d_91_conv2d_readvariableop_resource: @7
)conv2d_91_biasadd_readvariableop_resource:@B
(conv2d_92_conv2d_readvariableop_resource:@@7
)conv2d_92_biasadd_readvariableop_resource:@B
(conv2d_93_conv2d_readvariableop_resource:@@7
)conv2d_93_biasadd_readvariableop_resource:@B
(conv2d_94_conv2d_readvariableop_resource:@@7
)conv2d_94_biasadd_readvariableop_resource:@B
(conv2d_95_conv2d_readvariableop_resource:@@7
)conv2d_95_biasadd_readvariableop_resource:@:
'dense_30_matmul_readvariableop_resource:	@6
(dense_30_biasadd_readvariableop_resource:@9
'dense_31_matmul_readvariableop_resource:@6
(dense_31_biasadd_readvariableop_resource:
identity¢ conv2d_90/BiasAdd/ReadVariableOp¢conv2d_90/Conv2D/ReadVariableOp¢ conv2d_91/BiasAdd/ReadVariableOp¢conv2d_91/Conv2D/ReadVariableOp¢ conv2d_92/BiasAdd/ReadVariableOp¢conv2d_92/Conv2D/ReadVariableOp¢ conv2d_93/BiasAdd/ReadVariableOp¢conv2d_93/Conv2D/ReadVariableOp¢ conv2d_94/BiasAdd/ReadVariableOp¢conv2d_94/Conv2D/ReadVariableOp¢ conv2d_95/BiasAdd/ReadVariableOp¢conv2d_95/Conv2D/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢dense_31/MatMul/ReadVariableOp¢Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip¢Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip¢hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter¢<sequential_1/random_rotation/stateful_uniform/RngReadAndSkip
sequential/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
sequential/rescaling_1/Cast/x
sequential/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/rescaling_1/Cast_1/x«
sequential/rescaling_1/mulMulinputs&sequential/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/mulÇ
sequential/rescaling_1/addAddV2sequential/rescaling_1/mul:z:0(sequential/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/add
sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2!
sequential/resizing/resize/size
)sequential/resizing/resize/ResizeBilinearResizeBilinearsequential/rescaling_1/add:z:0(sequential/resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2+
)sequential/resizing/resize/ResizeBilinear¾
8sequential_1/random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_1/random_flip/stateful_uniform_full_int/shape¾
8sequential_1/random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_1/random_flip/stateful_uniform_full_int/Const¡
7sequential_1/random_flip/stateful_uniform_full_int/ProdProdAsequential_1/random_flip/stateful_uniform_full_int/shape:output:0Asequential_1/random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 29
7sequential_1/random_flip/stateful_uniform_full_int/Prod¸
9sequential_1/random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2;
9sequential_1/random_flip/stateful_uniform_full_int/Cast/xð
9sequential_1/random_flip/stateful_uniform_full_int/Cast_1Cast@sequential_1/random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2;
9sequential_1/random_flip/stateful_uniform_full_int/Cast_1
Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkipJsequential_1_random_flip_stateful_uniform_full_int_rngreadandskip_resourceBsequential_1/random_flip/stateful_uniform_full_int/Cast/x:output:0=sequential_1/random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2C
Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipÚ
Fsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2H
Fsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stackÞ
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_1Þ
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_2
@sequential_1/random_flip/stateful_uniform_full_int/strided_sliceStridedSliceIsequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Osequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack:output:0Qsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Qsequential_1/random_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2B
@sequential_1/random_flip/stateful_uniform_full_int/strided_sliceÿ
:sequential_1/random_flip/stateful_uniform_full_int/BitcastBitcastIsequential_1/random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02<
:sequential_1/random_flip/stateful_uniform_full_int/BitcastÞ
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stackâ
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1â
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2
Bsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1StridedSliceIsequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Qsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Ssequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Ssequential_1/random_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2D
Bsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1
<sequential_1/random_flip/stateful_uniform_full_int/Bitcast_1BitcastKsequential_1/random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02>
<sequential_1/random_flip/stateful_uniform_full_int/Bitcast_1²
6sequential_1/random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_1/random_flip/stateful_uniform_full_int/algÄ
2sequential_1/random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV2Asequential_1/random_flip/stateful_uniform_full_int/shape:output:0Esequential_1/random_flip/stateful_uniform_full_int/Bitcast_1:output:0Csequential_1/random_flip/stateful_uniform_full_int/Bitcast:output:0?sequential_1/random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	24
2sequential_1/random_flip/stateful_uniform_full_int
#sequential_1/random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2%
#sequential_1/random_flip/zeros_likeå
sequential_1/random_flip/stackPack;sequential_1/random_flip/stateful_uniform_full_int:output:0,sequential_1/random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:2 
sequential_1/random_flip/stack­
,sequential_1/random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2.
,sequential_1/random_flip/strided_slice/stack±
.sequential_1/random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.sequential_1/random_flip/strided_slice/stack_1±
.sequential_1/random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      20
.sequential_1/random_flip/strided_slice/stack_2
&sequential_1/random_flip/strided_sliceStridedSlice'sequential_1/random_flip/stack:output:05sequential_1/random_flip/strided_slice/stack:output:07sequential_1/random_flip/strided_slice/stack_1:output:07sequential_1/random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2(
&sequential_1/random_flip/strided_sliceÞ
Lsequential_1/random_flip/stateless_random_flip_left_right/control_dependencyIdentity:sequential/resizing/resize/ResizeBilinear:resized_images:0*
T0*<
_class2
0.loc:@sequential/resizing/resize/ResizeBilinear*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2N
Lsequential_1/random_flip/stateless_random_flip_left_right/control_dependency
?sequential_1/random_flip/stateless_random_flip_left_right/ShapeShapeUsequential_1/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2A
?sequential_1/random_flip/stateless_random_flip_left_right/Shapeè
Msequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2O
Msequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stackì
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_1ì
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_2¾
Gsequential_1/random_flip/stateless_random_flip_left_right/strided_sliceStridedSliceHsequential_1/random_flip/stateless_random_flip_left_right/Shape:output:0Vsequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Xsequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Xsequential_1/random_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2I
Gsequential_1/random_flip/stateless_random_flip_left_right/strided_slice¼
Xsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapePackPsequential_1/random_flip/stateless_random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2Z
Xsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeõ
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/minõ
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/maxÕ
osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter/sequential_1/random_flip/strided_slice:output:0* 
_output_shapes
::2q
osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter÷
hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgp^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2j
hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgà
ksequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2asequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0usequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ysequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0nsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2m
ksequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/subSub_sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0_sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub·
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMultsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Zsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2X
Vsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul
Rsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Zsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0_sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2T
Rsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniformØ
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/1Ø
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/2Ø
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2K
Isequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/3
Gsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shapePackPsequential_1/random_flip/stateless_random_flip_left_right/strided_slice:output:0Rsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/1:output:0Rsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/2:output:0Rsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2I
Gsequential_1/random_flip/stateless_random_flip_left_right/Reshape/shapeõ
Asequential_1/random_flip/stateless_random_flip_left_right/ReshapeReshapeVsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0Psequential_1/random_flip/stateless_random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2C
Asequential_1/random_flip/stateless_random_flip_left_right/Reshape
?sequential_1/random_flip/stateless_random_flip_left_right/RoundRoundJsequential_1/random_flip/stateless_random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?sequential_1/random_flip/stateless_random_flip_left_right/RoundÞ
Hsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2J
Hsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2/axisý
Csequential_1/random_flip/stateless_random_flip_left_right/ReverseV2	ReverseV2Usequential_1/random_flip/stateless_random_flip_left_right/control_dependency:output:0Qsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2E
Csequential_1/random_flip/stateless_random_flip_left_right/ReverseV2Ô
=sequential_1/random_flip/stateless_random_flip_left_right/mulMulCsequential_1/random_flip/stateless_random_flip_left_right/Round:y:0Lsequential_1/random_flip/stateless_random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=sequential_1/random_flip/stateless_random_flip_left_right/mulÇ
?sequential_1/random_flip/stateless_random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2A
?sequential_1/random_flip/stateless_random_flip_left_right/sub/xÎ
=sequential_1/random_flip/stateless_random_flip_left_right/subSubHsequential_1/random_flip/stateless_random_flip_left_right/sub/x:output:0Csequential_1/random_flip/stateless_random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=sequential_1/random_flip/stateless_random_flip_left_right/subß
?sequential_1/random_flip/stateless_random_flip_left_right/mul_1MulAsequential_1/random_flip/stateless_random_flip_left_right/sub:z:0Usequential_1/random_flip/stateless_random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?sequential_1/random_flip/stateless_random_flip_left_right/mul_1Ë
=sequential_1/random_flip/stateless_random_flip_left_right/addAddV2Asequential_1/random_flip/stateless_random_flip_left_right/mul:z:0Csequential_1/random_flip/stateless_random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=sequential_1/random_flip/stateless_random_flip_left_right/addÂ
:sequential_1/random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/random_flip/stateful_uniform_full_int_1/shapeÂ
:sequential_1/random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential_1/random_flip/stateful_uniform_full_int_1/Const©
9sequential_1/random_flip/stateful_uniform_full_int_1/ProdProdCsequential_1/random_flip/stateful_uniform_full_int_1/shape:output:0Csequential_1/random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2;
9sequential_1/random_flip/stateful_uniform_full_int_1/Prod¼
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast/xö
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast_1CastBsequential_1/random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2=
;sequential_1/random_flip/stateful_uniform_full_int_1/Cast_1Ï
Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkipJsequential_1_random_flip_stateful_uniform_full_int_rngreadandskip_resourceDsequential_1/random_flip/stateful_uniform_full_int_1/Cast/x:output:0?sequential_1/random_flip/stateful_uniform_full_int_1/Cast_1:y:0B^sequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:2E
Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipÞ
Hsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2J
Hsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stackâ
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1â
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2¦
Bsequential_1/random_flip/stateful_uniform_full_int_1/strided_sliceStridedSliceKsequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Qsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Ssequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Ssequential_1/random_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2D
Bsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice
<sequential_1/random_flip/stateful_uniform_full_int_1/BitcastBitcastKsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02>
<sequential_1/random_flip/stateful_uniform_full_int_1/Bitcastâ
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2L
Jsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stackæ
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1æ
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2
Dsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSliceKsequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Ssequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Usequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Usequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2F
Dsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1
>sequential_1/random_flip/stateful_uniform_full_int_1/Bitcast_1BitcastMsequential_1/random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02@
>sequential_1/random_flip/stateful_uniform_full_int_1/Bitcast_1¶
8sequential_1/random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_1/random_flip/stateful_uniform_full_int_1/algÐ
4sequential_1/random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV2Csequential_1/random_flip/stateful_uniform_full_int_1/shape:output:0Gsequential_1/random_flip/stateful_uniform_full_int_1/Bitcast_1:output:0Esequential_1/random_flip/stateful_uniform_full_int_1/Bitcast:output:0Asequential_1/random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	26
4sequential_1/random_flip/stateful_uniform_full_int_1
%sequential_1/random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2'
%sequential_1/random_flip/zeros_like_1í
 sequential_1/random_flip/stack_1Pack=sequential_1/random_flip/stateful_uniform_full_int_1:output:0.sequential_1/random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2"
 sequential_1/random_flip/stack_1±
.sequential_1/random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        20
.sequential_1/random_flip/strided_slice_1/stackµ
0sequential_1/random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0sequential_1/random_flip/strided_slice_1/stack_1µ
0sequential_1/random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0sequential_1/random_flip/strided_slice_1/stack_2ª
(sequential_1/random_flip/strided_slice_1StridedSlice)sequential_1/random_flip/stack_1:output:07sequential_1/random_flip/strided_slice_1/stack:output:09sequential_1/random_flip/strided_slice_1/stack_1:output:09sequential_1/random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2*
(sequential_1/random_flip/strided_slice_1ó
Isequential_1/random_flip/stateless_random_flip_up_down/control_dependencyIdentityAsequential_1/random_flip/stateless_random_flip_left_right/add:z:0*
T0*P
_classF
DBloc:@sequential_1/random_flip/stateless_random_flip_left_right/add*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Isequential_1/random_flip/stateless_random_flip_up_down/control_dependencyþ
<sequential_1/random_flip/stateless_random_flip_up_down/ShapeShapeRsequential_1/random_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2>
<sequential_1/random_flip/stateless_random_flip_up_down/Shapeâ
Jsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2L
Jsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stackæ
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_1æ
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2N
Lsequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_2¬
Dsequential_1/random_flip/stateless_random_flip_up_down/strided_sliceStridedSliceEsequential_1/random_flip/stateless_random_flip_up_down/Shape:output:0Ssequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Usequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Usequential_1/random_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2F
Dsequential_1/random_flip/stateless_random_flip_up_down/strided_slice³
Usequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shapePackMsequential_1/random_flip/stateless_random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2W
Usequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeï
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/minï
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/max¼
lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter1sequential_1/random_flip/strided_slice_1:output:0i^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg* 
_output_shapes
::2n
lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterî
esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgm^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2g
esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgÎ
hsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0rsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0vsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0ksequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2j
hsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/subSub\sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0\sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/sub«
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMulqsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Wsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2U
Ssequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mul
Osequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Wsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0\sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2Q
Osequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniformÒ
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/1Ò
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/2Ò
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2H
Fsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/3
Dsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shapePackMsequential_1/random_flip/stateless_random_flip_up_down/strided_slice:output:0Osequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/1:output:0Osequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/2:output:0Osequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2F
Dsequential_1/random_flip/stateless_random_flip_up_down/Reshape/shapeé
>sequential_1/random_flip/stateless_random_flip_up_down/ReshapeReshapeSsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform:z:0Msequential_1/random_flip/stateless_random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2@
>sequential_1/random_flip/stateless_random_flip_up_down/Reshape
<sequential_1/random_flip/stateless_random_flip_up_down/RoundRoundGsequential_1/random_flip/stateless_random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<sequential_1/random_flip/stateless_random_flip_up_down/RoundØ
Esequential_1/random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_1/random_flip/stateless_random_flip_up_down/ReverseV2/axisñ
@sequential_1/random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2Rsequential_1/random_flip/stateless_random_flip_up_down/control_dependency:output:0Nsequential_1/random_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2B
@sequential_1/random_flip/stateless_random_flip_up_down/ReverseV2È
:sequential_1/random_flip/stateless_random_flip_up_down/mulMul@sequential_1/random_flip/stateless_random_flip_up_down/Round:y:0Isequential_1/random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_1/random_flip/stateless_random_flip_up_down/mulÁ
<sequential_1/random_flip/stateless_random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2>
<sequential_1/random_flip/stateless_random_flip_up_down/sub/xÂ
:sequential_1/random_flip/stateless_random_flip_up_down/subSubEsequential_1/random_flip/stateless_random_flip_up_down/sub/x:output:0@sequential_1/random_flip/stateless_random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_1/random_flip/stateless_random_flip_up_down/subÓ
<sequential_1/random_flip/stateless_random_flip_up_down/mul_1Mul>sequential_1/random_flip/stateless_random_flip_up_down/sub:z:0Rsequential_1/random_flip/stateless_random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<sequential_1/random_flip/stateless_random_flip_up_down/mul_1¿
:sequential_1/random_flip/stateless_random_flip_up_down/addAddV2>sequential_1/random_flip/stateless_random_flip_up_down/mul:z:0@sequential_1/random_flip/stateless_random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:sequential_1/random_flip/stateless_random_flip_up_down/add¶
"sequential_1/random_rotation/ShapeShape>sequential_1/random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2$
"sequential_1/random_rotation/Shape®
0sequential_1/random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0sequential_1/random_rotation/strided_slice/stack²
2sequential_1/random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_rotation/strided_slice/stack_1²
2sequential_1/random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2sequential_1/random_rotation/strided_slice/stack_2
*sequential_1/random_rotation/strided_sliceStridedSlice+sequential_1/random_rotation/Shape:output:09sequential_1/random_rotation/strided_slice/stack:output:0;sequential_1/random_rotation/strided_slice/stack_1:output:0;sequential_1/random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*sequential_1/random_rotation/strided_slice»
2sequential_1/random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/strided_slice_1/stack¿
4sequential_1/random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ26
4sequential_1/random_rotation/strided_slice_1/stack_1¶
4sequential_1/random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation/strided_slice_1/stack_2
,sequential_1/random_rotation/strided_slice_1StridedSlice+sequential_1/random_rotation/Shape:output:0;sequential_1/random_rotation/strided_slice_1/stack:output:0=sequential_1/random_rotation/strided_slice_1/stack_1:output:0=sequential_1/random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_1/random_rotation/strided_slice_1µ
!sequential_1/random_rotation/CastCast5sequential_1/random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2#
!sequential_1/random_rotation/Cast»
2sequential_1/random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/strided_slice_2/stack¿
4sequential_1/random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ26
4sequential_1/random_rotation/strided_slice_2/stack_1¶
4sequential_1/random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_1/random_rotation/strided_slice_2/stack_2
,sequential_1/random_rotation/strided_slice_2StridedSlice+sequential_1/random_rotation/Shape:output:0;sequential_1/random_rotation/strided_slice_2/stack:output:0=sequential_1/random_rotation/strided_slice_2/stack_1:output:0=sequential_1/random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_1/random_rotation/strided_slice_2¹
#sequential_1/random_rotation/Cast_1Cast5sequential_1/random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_1/random_rotation/Cast_1Õ
3sequential_1/random_rotation/stateful_uniform/shapePack3sequential_1/random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:25
3sequential_1/random_rotation/stateful_uniform/shape«
1sequential_1/random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿23
1sequential_1/random_rotation/stateful_uniform/min«
1sequential_1/random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?23
1sequential_1/random_rotation/stateful_uniform/max´
3sequential_1/random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential_1/random_rotation/stateful_uniform/Const
2sequential_1/random_rotation/stateful_uniform/ProdProd<sequential_1/random_rotation/stateful_uniform/shape:output:0<sequential_1/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/stateful_uniform/Prod®
4sequential_1/random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :26
4sequential_1/random_rotation/stateful_uniform/Cast/xá
4sequential_1/random_rotation/stateful_uniform/Cast_1Cast;sequential_1/random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 26
4sequential_1/random_rotation/stateful_uniform/Cast_1ê
<sequential_1/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkipEsequential_1_random_rotation_stateful_uniform_rngreadandskip_resource=sequential_1/random_rotation/stateful_uniform/Cast/x:output:08sequential_1/random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:2>
<sequential_1/random_rotation/stateful_uniform/RngReadAndSkipÐ
Asequential_1/random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Asequential_1/random_rotation/stateful_uniform/strided_slice/stackÔ
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_1Ô
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_1/random_rotation/stateful_uniform/strided_slice/stack_2ü
;sequential_1/random_rotation/stateful_uniform/strided_sliceStridedSliceDsequential_1/random_rotation/stateful_uniform/RngReadAndSkip:value:0Jsequential_1/random_rotation/stateful_uniform/strided_slice/stack:output:0Lsequential_1/random_rotation/stateful_uniform/strided_slice/stack_1:output:0Lsequential_1/random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2=
;sequential_1/random_rotation/stateful_uniform/strided_sliceð
5sequential_1/random_rotation/stateful_uniform/BitcastBitcastDsequential_1/random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type027
5sequential_1/random_rotation/stateful_uniform/BitcastÔ
Csequential_1/random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2E
Csequential_1/random_rotation/stateful_uniform/strided_slice_1/stackØ
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_1Ø
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_2ô
=sequential_1/random_rotation/stateful_uniform/strided_slice_1StridedSliceDsequential_1/random_rotation/stateful_uniform/RngReadAndSkip:value:0Lsequential_1/random_rotation/stateful_uniform/strided_slice_1/stack:output:0Nsequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Nsequential_1/random_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2?
=sequential_1/random_rotation/stateful_uniform/strided_slice_1ö
7sequential_1/random_rotation/stateful_uniform/Bitcast_1BitcastFsequential_1/random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type029
7sequential_1/random_rotation/stateful_uniform/Bitcast_1Ú
Jsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2L
Jsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2/algæ
Fsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2<sequential_1/random_rotation/stateful_uniform/shape:output:0@sequential_1/random_rotation/stateful_uniform/Bitcast_1:output:0>sequential_1/random_rotation/stateful_uniform/Bitcast:output:0Ssequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2H
Fsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2
1sequential_1/random_rotation/stateful_uniform/subSub:sequential_1/random_rotation/stateful_uniform/max:output:0:sequential_1/random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 23
1sequential_1/random_rotation/stateful_uniform/sub£
1sequential_1/random_rotation/stateful_uniform/mulMulOsequential_1/random_rotation/stateful_uniform/StatelessRandomUniformV2:output:05sequential_1/random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ23
1sequential_1/random_rotation/stateful_uniform/mul
-sequential_1/random_rotation/stateful_uniformAddV25sequential_1/random_rotation/stateful_uniform/mul:z:0:sequential_1/random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-sequential_1/random_rotation/stateful_uniform­
2sequential_1/random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?24
2sequential_1/random_rotation/rotation_matrix/sub/yò
0sequential_1/random_rotation/rotation_matrix/subSub'sequential_1/random_rotation/Cast_1:y:0;sequential_1/random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 22
0sequential_1/random_rotation/rotation_matrix/subÌ
0sequential_1/random_rotation/rotation_matrix/CosCos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/Cos±
4sequential_1/random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_1/yø
2sequential_1/random_rotation/rotation_matrix/sub_1Sub'sequential_1/random_rotation/Cast_1:y:0=sequential_1/random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_1
0sequential_1/random_rotation/rotation_matrix/mulMul4sequential_1/random_rotation/rotation_matrix/Cos:y:06sequential_1/random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/mulÌ
0sequential_1/random_rotation/rotation_matrix/SinSin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/Sin±
4sequential_1/random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_2/yö
2sequential_1/random_rotation/rotation_matrix/sub_2Sub%sequential_1/random_rotation/Cast:y:0=sequential_1/random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_2
2sequential_1/random_rotation/rotation_matrix/mul_1Mul4sequential_1/random_rotation/rotation_matrix/Sin:y:06sequential_1/random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/mul_1
2sequential_1/random_rotation/rotation_matrix/sub_3Sub4sequential_1/random_rotation/rotation_matrix/mul:z:06sequential_1/random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/sub_3
2sequential_1/random_rotation/rotation_matrix/sub_4Sub4sequential_1/random_rotation/rotation_matrix/sub:z:06sequential_1/random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/sub_4µ
6sequential_1/random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @28
6sequential_1/random_rotation/rotation_matrix/truediv/y
4sequential_1/random_rotation/rotation_matrix/truedivRealDiv6sequential_1/random_rotation/rotation_matrix/sub_4:z:0?sequential_1/random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4sequential_1/random_rotation/rotation_matrix/truediv±
4sequential_1/random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_5/yö
2sequential_1/random_rotation/rotation_matrix/sub_5Sub%sequential_1/random_rotation/Cast:y:0=sequential_1/random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_5Ð
2sequential_1/random_rotation/rotation_matrix/Sin_1Sin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Sin_1±
4sequential_1/random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_6/yø
2sequential_1/random_rotation/rotation_matrix/sub_6Sub'sequential_1/random_rotation/Cast_1:y:0=sequential_1/random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_6
2sequential_1/random_rotation/rotation_matrix/mul_2Mul6sequential_1/random_rotation/rotation_matrix/Sin_1:y:06sequential_1/random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/mul_2Ð
2sequential_1/random_rotation/rotation_matrix/Cos_1Cos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Cos_1±
4sequential_1/random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?26
4sequential_1/random_rotation/rotation_matrix/sub_7/yö
2sequential_1/random_rotation/rotation_matrix/sub_7Sub%sequential_1/random_rotation/Cast:y:0=sequential_1/random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 24
2sequential_1/random_rotation/rotation_matrix/sub_7
2sequential_1/random_rotation/rotation_matrix/mul_3Mul6sequential_1/random_rotation/rotation_matrix/Cos_1:y:06sequential_1/random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/mul_3
0sequential_1/random_rotation/rotation_matrix/addAddV26sequential_1/random_rotation/rotation_matrix/mul_2:z:06sequential_1/random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/add
2sequential_1/random_rotation/rotation_matrix/sub_8Sub6sequential_1/random_rotation/rotation_matrix/sub_5:z:04sequential_1/random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/sub_8¹
8sequential_1/random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2:
8sequential_1/random_rotation/rotation_matrix/truediv_1/y¤
6sequential_1/random_rotation/rotation_matrix/truediv_1RealDiv6sequential_1/random_rotation/rotation_matrix/sub_8:z:0Asequential_1/random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6sequential_1/random_rotation/rotation_matrix/truediv_1É
2sequential_1/random_rotation/rotation_matrix/ShapeShape1sequential_1/random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:24
2sequential_1/random_rotation/rotation_matrix/ShapeÎ
@sequential_1/random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2B
@sequential_1/random_rotation/rotation_matrix/strided_slice/stackÒ
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_1Ò
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice/stack_2ð
:sequential_1/random_rotation/rotation_matrix/strided_sliceStridedSlice;sequential_1/random_rotation/rotation_matrix/Shape:output:0Isequential_1/random_rotation/rotation_matrix/strided_slice/stack:output:0Ksequential_1/random_rotation/rotation_matrix/strided_slice/stack_1:output:0Ksequential_1/random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2<
:sequential_1/random_rotation/rotation_matrix/strided_sliceÐ
2sequential_1/random_rotation/rotation_matrix/Cos_2Cos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Cos_2Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_1/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_1StridedSlice6sequential_1/random_rotation/rotation_matrix/Cos_2:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_1/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_1Ð
2sequential_1/random_rotation/rotation_matrix/Sin_2Sin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Sin_2Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_2/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_2StridedSlice6sequential_1/random_rotation/rotation_matrix/Sin_2:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_2/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_2ä
0sequential_1/random_rotation/rotation_matrix/NegNegEsequential_1/random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0sequential_1/random_rotation/rotation_matrix/NegÙ
Bsequential_1/random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_3/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_2§
<sequential_1/random_rotation/rotation_matrix/strided_slice_3StridedSlice8sequential_1/random_rotation/rotation_matrix/truediv:z:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_3/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_3Ð
2sequential_1/random_rotation/rotation_matrix/Sin_3Sin1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Sin_3Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_4/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_4StridedSlice6sequential_1/random_rotation/rotation_matrix/Sin_3:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_4/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_4Ð
2sequential_1/random_rotation/rotation_matrix/Cos_3Cos1sequential_1/random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/Cos_3Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_5/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_2¥
<sequential_1/random_rotation/rotation_matrix/strided_slice_5StridedSlice6sequential_1/random_rotation/rotation_matrix/Cos_3:y:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_5/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_5Ù
Bsequential_1/random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2D
Bsequential_1/random_rotation/rotation_matrix/strided_slice_6/stackÝ
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_1Ý
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2F
Dsequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_2©
<sequential_1/random_rotation/rotation_matrix/strided_slice_6StridedSlice:sequential_1/random_rotation/rotation_matrix/truediv_1:z:0Ksequential_1/random_rotation/rotation_matrix/strided_slice_6/stack:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0Msequential_1/random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2>
<sequential_1/random_rotation/rotation_matrix/strided_slice_6¼
;sequential_1/random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2=
;sequential_1/random_rotation/rotation_matrix/zeros/packed/1·
9sequential_1/random_rotation/rotation_matrix/zeros/packedPackCsequential_1/random_rotation/rotation_matrix/strided_slice:output:0Dsequential_1/random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2;
9sequential_1/random_rotation/rotation_matrix/zeros/packed¹
8sequential_1/random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2:
8sequential_1/random_rotation/rotation_matrix/zeros/Const©
2sequential_1/random_rotation/rotation_matrix/zerosFillBsequential_1/random_rotation/rotation_matrix/zeros/packed:output:0Asequential_1/random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2sequential_1/random_rotation/rotation_matrix/zeros¶
8sequential_1/random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_1/random_rotation/rotation_matrix/concat/axisÊ
3sequential_1/random_rotation/rotation_matrix/concatConcatV2Esequential_1/random_rotation/rotation_matrix/strided_slice_1:output:04sequential_1/random_rotation/rotation_matrix/Neg:y:0Esequential_1/random_rotation/rotation_matrix/strided_slice_3:output:0Esequential_1/random_rotation/rotation_matrix/strided_slice_4:output:0Esequential_1/random_rotation/rotation_matrix/strided_slice_5:output:0Esequential_1/random_rotation/rotation_matrix/strided_slice_6:output:0;sequential_1/random_rotation/rotation_matrix/zeros:output:0Asequential_1/random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3sequential_1/random_rotation/rotation_matrix/concatÊ
,sequential_1/random_rotation/transform/ShapeShape>sequential_1/random_flip/stateless_random_flip_up_down/add:z:0*
T0*
_output_shapes
:2.
,sequential_1/random_rotation/transform/ShapeÂ
:sequential_1/random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_1/random_rotation/transform/strided_slice/stackÆ
<sequential_1/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_rotation/transform/strided_slice/stack_1Æ
<sequential_1/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_1/random_rotation/transform/strided_slice/stack_2¸
4sequential_1/random_rotation/transform/strided_sliceStridedSlice5sequential_1/random_rotation/transform/Shape:output:0Csequential_1/random_rotation/transform/strided_slice/stack:output:0Esequential_1/random_rotation/transform/strided_slice/stack_1:output:0Esequential_1/random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:26
4sequential_1/random_rotation/transform/strided_slice«
1sequential_1/random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential_1/random_rotation/transform/fill_value
Asequential_1/random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3>sequential_1/random_flip/stateless_random_flip_up_down/add:z:0<sequential_1/random_rotation/rotation_matrix/concat:output:0=sequential_1/random_rotation/transform/strided_slice:output:0:sequential_1/random_rotation/transform/fill_value:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2C
Asequential_1/random_rotation/transform/ImageProjectiveTransformV3³
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_90/Conv2D/ReadVariableOp
conv2d_90/Conv2DConv2DVsequential_1/random_rotation/transform/ImageProjectiveTransformV3:transformed_images:0'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2
conv2d_90/Conv2Dª
 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_90/BiasAdd/ReadVariableOp²
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/BiasAdd
conv2d_90/ReluReluconv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/ReluÊ
max_pooling2d_90/MaxPoolMaxPoolconv2d_90/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_90/MaxPool³
conv2d_91/Conv2D/ReadVariableOpReadVariableOp(conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_91/Conv2D/ReadVariableOpÝ
conv2d_91/Conv2DConv2D!max_pooling2d_90/MaxPool:output:0'conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2
conv2d_91/Conv2Dª
 conv2d_91/BiasAdd/ReadVariableOpReadVariableOp)conv2d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_91/BiasAdd/ReadVariableOp°
conv2d_91/BiasAddBiasAddconv2d_91/Conv2D:output:0(conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/BiasAdd~
conv2d_91/ReluReluconv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/ReluÊ
max_pooling2d_91/MaxPoolMaxPoolconv2d_91/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_91/MaxPool³
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_92/Conv2D/ReadVariableOpÝ
conv2d_92/Conv2DConv2D!max_pooling2d_91/MaxPool:output:0'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2
conv2d_92/Conv2Dª
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp°
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/ReluÊ
max_pooling2d_92/MaxPoolMaxPoolconv2d_92/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_92/MaxPool³
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_93/Conv2D/ReadVariableOpÝ
conv2d_93/Conv2DConv2D!max_pooling2d_92/MaxPool:output:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_93/Conv2Dª
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp°
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/ReluÊ
max_pooling2d_93/MaxPoolMaxPoolconv2d_93/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_93/MaxPool³
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_94/Conv2D/ReadVariableOpÝ
conv2d_94/Conv2DConv2D!max_pooling2d_93/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_94/Conv2Dª
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp°
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/ReluÊ
max_pooling2d_94/MaxPoolMaxPoolconv2d_94/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_94/MaxPool³
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_95/Conv2D/ReadVariableOpÝ
conv2d_95/Conv2DConv2D!max_pooling2d_94/MaxPool:output:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_95/Conv2Dª
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp°
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/ReluÊ
max_pooling2d_95/MaxPoolMaxPoolconv2d_95/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_95/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_15/Const¤
flatten_15/ReshapeReshape!max_pooling2d_95/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_15/Reshape©
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_30/MatMul/ReadVariableOp£
dense_30/MatMulMatMulflatten_15/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/MatMul§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp¥
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_31/MatMul/ReadVariableOp£
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMul§
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp¥
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/BiasAdd|
dense_31/SoftmaxSoftmaxdense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/Softmaxu
IdentityIdentitydense_31/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityï	
NoOpNoOp!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp!^conv2d_91/BiasAdd/ReadVariableOp ^conv2d_91/Conv2D/ReadVariableOp!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOpB^sequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipD^sequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipi^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgp^sequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterf^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgm^sequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter=^sequential_1/random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp2D
 conv2d_91/BiasAdd/ReadVariableOp conv2d_91/BiasAdd/ReadVariableOp2B
conv2d_91/Conv2D/ReadVariableOpconv2d_91/Conv2D/ReadVariableOp2D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp2
Asequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkipAsequential_1/random_flip/stateful_uniform_full_int/RngReadAndSkip2
Csequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkipCsequential_1/random_flip/stateful_uniform_full_int_1/RngReadAndSkip2Ô
hsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlghsequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2â
osequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterosequential_1/random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2Î
esequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgesequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2Ü
lsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterlsequential_1/random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter2|
<sequential_1/random_rotation/stateful_uniform/RngReadAndSkip<sequential_1/random_rotation/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ë
.__inference_sequential_17_layer_call_fn_162261

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@

unknown_11:	@

unknown_12:@

unknown_13:@

unknown_14:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1614462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

!__inference__wrapped_model_160644
sequential_inputP
6sequential_17_conv2d_90_conv2d_readvariableop_resource: E
7sequential_17_conv2d_90_biasadd_readvariableop_resource: P
6sequential_17_conv2d_91_conv2d_readvariableop_resource: @E
7sequential_17_conv2d_91_biasadd_readvariableop_resource:@P
6sequential_17_conv2d_92_conv2d_readvariableop_resource:@@E
7sequential_17_conv2d_92_biasadd_readvariableop_resource:@P
6sequential_17_conv2d_93_conv2d_readvariableop_resource:@@E
7sequential_17_conv2d_93_biasadd_readvariableop_resource:@P
6sequential_17_conv2d_94_conv2d_readvariableop_resource:@@E
7sequential_17_conv2d_94_biasadd_readvariableop_resource:@P
6sequential_17_conv2d_95_conv2d_readvariableop_resource:@@E
7sequential_17_conv2d_95_biasadd_readvariableop_resource:@H
5sequential_17_dense_30_matmul_readvariableop_resource:	@D
6sequential_17_dense_30_biasadd_readvariableop_resource:@G
5sequential_17_dense_31_matmul_readvariableop_resource:@D
6sequential_17_dense_31_biasadd_readvariableop_resource:
identity¢.sequential_17/conv2d_90/BiasAdd/ReadVariableOp¢-sequential_17/conv2d_90/Conv2D/ReadVariableOp¢.sequential_17/conv2d_91/BiasAdd/ReadVariableOp¢-sequential_17/conv2d_91/Conv2D/ReadVariableOp¢.sequential_17/conv2d_92/BiasAdd/ReadVariableOp¢-sequential_17/conv2d_92/Conv2D/ReadVariableOp¢.sequential_17/conv2d_93/BiasAdd/ReadVariableOp¢-sequential_17/conv2d_93/Conv2D/ReadVariableOp¢.sequential_17/conv2d_94/BiasAdd/ReadVariableOp¢-sequential_17/conv2d_94/Conv2D/ReadVariableOp¢.sequential_17/conv2d_95/BiasAdd/ReadVariableOp¢-sequential_17/conv2d_95/Conv2D/ReadVariableOp¢-sequential_17/dense_30/BiasAdd/ReadVariableOp¢,sequential_17/dense_30/MatMul/ReadVariableOp¢-sequential_17/dense_31/BiasAdd/ReadVariableOp¢,sequential_17/dense_31/MatMul/ReadVariableOp
+sequential_17/sequential/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2-
+sequential_17/sequential/rescaling_1/Cast/x£
-sequential_17/sequential/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-sequential_17/sequential/rescaling_1/Cast_1/xß
(sequential_17/sequential/rescaling_1/mulMulsequential_input4sequential_17/sequential/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_17/sequential/rescaling_1/mulÿ
(sequential_17/sequential/rescaling_1/addAddV2,sequential_17/sequential/rescaling_1/mul:z:06sequential_17/sequential/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(sequential_17/sequential/rescaling_1/add¯
-sequential_17/sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2/
-sequential_17/sequential/resizing/resize/sizeÀ
7sequential_17/sequential/resizing/resize/ResizeBilinearResizeBilinear,sequential_17/sequential/rescaling_1/add:z:06sequential_17/sequential/resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(29
7sequential_17/sequential/resizing/resize/ResizeBilinearÝ
-sequential_17/conv2d_90/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02/
-sequential_17/conv2d_90/Conv2D/ReadVariableOp°
sequential_17/conv2d_90/Conv2DConv2DHsequential_17/sequential/resizing/resize/ResizeBilinear:resized_images:05sequential_17/conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2 
sequential_17/conv2d_90/Conv2DÔ
.sequential_17/conv2d_90/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.sequential_17/conv2d_90/BiasAdd/ReadVariableOpê
sequential_17/conv2d_90/BiasAddBiasAdd'sequential_17/conv2d_90/Conv2D:output:06sequential_17/conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2!
sequential_17/conv2d_90/BiasAddª
sequential_17/conv2d_90/ReluRelu(sequential_17/conv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
sequential_17/conv2d_90/Reluô
&sequential_17/max_pooling2d_90/MaxPoolMaxPool*sequential_17/conv2d_90/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_90/MaxPoolÝ
-sequential_17/conv2d_91/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-sequential_17/conv2d_91/Conv2D/ReadVariableOp
sequential_17/conv2d_91/Conv2DConv2D/sequential_17/max_pooling2d_90/MaxPool:output:05sequential_17/conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2 
sequential_17/conv2d_91/Conv2DÔ
.sequential_17/conv2d_91/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_91/BiasAdd/ReadVariableOpè
sequential_17/conv2d_91/BiasAddBiasAdd'sequential_17/conv2d_91/Conv2D:output:06sequential_17/conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2!
sequential_17/conv2d_91/BiasAdd¨
sequential_17/conv2d_91/ReluRelu(sequential_17/conv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
sequential_17/conv2d_91/Reluô
&sequential_17/max_pooling2d_91/MaxPoolMaxPool*sequential_17/conv2d_91/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_91/MaxPoolÝ
-sequential_17/conv2d_92/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_17/conv2d_92/Conv2D/ReadVariableOp
sequential_17/conv2d_92/Conv2DConv2D/sequential_17/max_pooling2d_91/MaxPool:output:05sequential_17/conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2 
sequential_17/conv2d_92/Conv2DÔ
.sequential_17/conv2d_92/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_92/BiasAdd/ReadVariableOpè
sequential_17/conv2d_92/BiasAddBiasAdd'sequential_17/conv2d_92/Conv2D:output:06sequential_17/conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2!
sequential_17/conv2d_92/BiasAdd¨
sequential_17/conv2d_92/ReluRelu(sequential_17/conv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
sequential_17/conv2d_92/Reluô
&sequential_17/max_pooling2d_92/MaxPoolMaxPool*sequential_17/conv2d_92/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_92/MaxPoolÝ
-sequential_17/conv2d_93/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_17/conv2d_93/Conv2D/ReadVariableOp
sequential_17/conv2d_93/Conv2DConv2D/sequential_17/max_pooling2d_92/MaxPool:output:05sequential_17/conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2 
sequential_17/conv2d_93/Conv2DÔ
.sequential_17/conv2d_93/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_93/BiasAdd/ReadVariableOpè
sequential_17/conv2d_93/BiasAddBiasAdd'sequential_17/conv2d_93/Conv2D:output:06sequential_17/conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_17/conv2d_93/BiasAdd¨
sequential_17/conv2d_93/ReluRelu(sequential_17/conv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_17/conv2d_93/Reluô
&sequential_17/max_pooling2d_93/MaxPoolMaxPool*sequential_17/conv2d_93/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_93/MaxPoolÝ
-sequential_17/conv2d_94/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_17/conv2d_94/Conv2D/ReadVariableOp
sequential_17/conv2d_94/Conv2DConv2D/sequential_17/max_pooling2d_93/MaxPool:output:05sequential_17/conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2 
sequential_17/conv2d_94/Conv2DÔ
.sequential_17/conv2d_94/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_94/BiasAdd/ReadVariableOpè
sequential_17/conv2d_94/BiasAddBiasAdd'sequential_17/conv2d_94/Conv2D:output:06sequential_17/conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_17/conv2d_94/BiasAdd¨
sequential_17/conv2d_94/ReluRelu(sequential_17/conv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_17/conv2d_94/Reluô
&sequential_17/max_pooling2d_94/MaxPoolMaxPool*sequential_17/conv2d_94/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_94/MaxPoolÝ
-sequential_17/conv2d_95/Conv2D/ReadVariableOpReadVariableOp6sequential_17_conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02/
-sequential_17/conv2d_95/Conv2D/ReadVariableOp
sequential_17/conv2d_95/Conv2DConv2D/sequential_17/max_pooling2d_94/MaxPool:output:05sequential_17/conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2 
sequential_17/conv2d_95/Conv2DÔ
.sequential_17/conv2d_95/BiasAdd/ReadVariableOpReadVariableOp7sequential_17_conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.sequential_17/conv2d_95/BiasAdd/ReadVariableOpè
sequential_17/conv2d_95/BiasAddBiasAdd'sequential_17/conv2d_95/Conv2D:output:06sequential_17/conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
sequential_17/conv2d_95/BiasAdd¨
sequential_17/conv2d_95/ReluRelu(sequential_17/conv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_17/conv2d_95/Reluô
&sequential_17/max_pooling2d_95/MaxPoolMaxPool*sequential_17/conv2d_95/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2(
&sequential_17/max_pooling2d_95/MaxPool
sequential_17/flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2 
sequential_17/flatten_15/ConstÜ
 sequential_17/flatten_15/ReshapeReshape/sequential_17/max_pooling2d_95/MaxPool:output:0'sequential_17/flatten_15/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 sequential_17/flatten_15/ReshapeÓ
,sequential_17/dense_30/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_30_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02.
,sequential_17/dense_30/MatMul/ReadVariableOpÛ
sequential_17/dense_30/MatMulMatMul)sequential_17/flatten_15/Reshape:output:04sequential_17/dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_17/dense_30/MatMulÑ
-sequential_17/dense_30/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02/
-sequential_17/dense_30/BiasAdd/ReadVariableOpÝ
sequential_17/dense_30/BiasAddBiasAdd'sequential_17/dense_30/MatMul:product:05sequential_17/dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
sequential_17/dense_30/BiasAdd
sequential_17/dense_30/ReluRelu'sequential_17/dense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
sequential_17/dense_30/ReluÒ
,sequential_17/dense_31/MatMul/ReadVariableOpReadVariableOp5sequential_17_dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential_17/dense_31/MatMul/ReadVariableOpÛ
sequential_17/dense_31/MatMulMatMul)sequential_17/dense_30/Relu:activations:04sequential_17/dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential_17/dense_31/MatMulÑ
-sequential_17/dense_31/BiasAdd/ReadVariableOpReadVariableOp6sequential_17_dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential_17/dense_31/BiasAdd/ReadVariableOpÝ
sequential_17/dense_31/BiasAddBiasAdd'sequential_17/dense_31/MatMul:product:05sequential_17/dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_17/dense_31/BiasAdd¦
sequential_17/dense_31/SoftmaxSoftmax'sequential_17/dense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
sequential_17/dense_31/Softmax
IdentityIdentity(sequential_17/dense_31/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÒ
NoOpNoOp/^sequential_17/conv2d_90/BiasAdd/ReadVariableOp.^sequential_17/conv2d_90/Conv2D/ReadVariableOp/^sequential_17/conv2d_91/BiasAdd/ReadVariableOp.^sequential_17/conv2d_91/Conv2D/ReadVariableOp/^sequential_17/conv2d_92/BiasAdd/ReadVariableOp.^sequential_17/conv2d_92/Conv2D/ReadVariableOp/^sequential_17/conv2d_93/BiasAdd/ReadVariableOp.^sequential_17/conv2d_93/Conv2D/ReadVariableOp/^sequential_17/conv2d_94/BiasAdd/ReadVariableOp.^sequential_17/conv2d_94/Conv2D/ReadVariableOp/^sequential_17/conv2d_95/BiasAdd/ReadVariableOp.^sequential_17/conv2d_95/Conv2D/ReadVariableOp.^sequential_17/dense_30/BiasAdd/ReadVariableOp-^sequential_17/dense_30/MatMul/ReadVariableOp.^sequential_17/dense_31/BiasAdd/ReadVariableOp-^sequential_17/dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2`
.sequential_17/conv2d_90/BiasAdd/ReadVariableOp.sequential_17/conv2d_90/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_90/Conv2D/ReadVariableOp-sequential_17/conv2d_90/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_91/BiasAdd/ReadVariableOp.sequential_17/conv2d_91/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_91/Conv2D/ReadVariableOp-sequential_17/conv2d_91/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_92/BiasAdd/ReadVariableOp.sequential_17/conv2d_92/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_92/Conv2D/ReadVariableOp-sequential_17/conv2d_92/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_93/BiasAdd/ReadVariableOp.sequential_17/conv2d_93/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_93/Conv2D/ReadVariableOp-sequential_17/conv2d_93/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_94/BiasAdd/ReadVariableOp.sequential_17/conv2d_94/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_94/Conv2D/ReadVariableOp-sequential_17/conv2d_94/Conv2D/ReadVariableOp2`
.sequential_17/conv2d_95/BiasAdd/ReadVariableOp.sequential_17/conv2d_95/BiasAdd/ReadVariableOp2^
-sequential_17/conv2d_95/Conv2D/ReadVariableOp-sequential_17/conv2d_95/Conv2D/ReadVariableOp2^
-sequential_17/dense_30/BiasAdd/ReadVariableOp-sequential_17/dense_30/BiasAdd/ReadVariableOp2\
,sequential_17/dense_30/MatMul/ReadVariableOp,sequential_17/dense_30/MatMul/ReadVariableOp2^
-sequential_17/dense_31/BiasAdd/ReadVariableOp-sequential_17/dense_31/BiasAdd/ReadVariableOp2\
,sequential_17/dense_31/MatMul/ReadVariableOp,sequential_17/dense_31/MatMul/ReadVariableOp:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namesequential_input
å
R
+__inference_sequential_layer_call_fn_163086
rescaling_1_input
identityÌ
PartitionedCallPartitionedCallrescaling_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1606742
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
+
_user_specified_namerescaling_1_input
ï
b
F__inference_sequential_layer_call_and_return_conditional_losses_163125

inputs
identityb
CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
Castm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/x
rescaling_1/mulMulCast:y:0rescaling_1/Cast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/mul
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
rescaling_1/add
resizing/resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
resizing/resize/ExpandDims/dim¿
resizing/resize/ExpandDims
ExpandDimsrescaling_1/add:z:0'resizing/resize/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
resizing/resize/ExpandDims}
resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing/resize/sizeã
resizing/resize/ResizeBilinearResizeBilinear#resizing/resize/ExpandDims:output:0resizing/resize/size:output:0*
T0*(
_output_shapes
:*
half_pixel_centers(2 
resizing/resize/ResizeBilinear´
resizing/resize/SqueezeSqueeze/resizing/resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 2
resizing/resize/Squeezeq
IdentityIdentity resizing/resize/Squeeze:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs


-__inference_sequential_1_layer_call_fn_163206

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1610782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
H
,__inference_random_flip_layer_call_fn_164343

inputs
identityÊ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_random_flip_layer_call_and_return_conditional_losses_1607382
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


*__inference_conv2d_91_layer_call_fn_164069

inputs!
unknown: @
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_91_layer_call_and_return_conditional_losses_1612992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs


-__inference_sequential_1_layer_call_fn_163229

inputs
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1618292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
ï
G__inference_random_flip_layer_call_and_return_conditional_losses_164511

inputs?
1stateful_uniform_full_int_rngreadandskip_resource:	
identity¢(stateful_uniform_full_int/RngReadAndSkip¢*stateful_uniform_full_int_1/RngReadAndSkip¢Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assert¢Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¢Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert¢Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert¢Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter
stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2!
stateful_uniform_full_int/shape
stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
stateful_uniform_full_int/Const½
stateful_uniform_full_int/ProdProd(stateful_uniform_full_int/shape:output:0(stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2 
stateful_uniform_full_int/Prod
 stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2"
 stateful_uniform_full_int/Cast/x¥
 stateful_uniform_full_int/Cast_1Cast'stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 stateful_uniform_full_int/Cast_1
(stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource)stateful_uniform_full_int/Cast/x:output:0$stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:2*
(stateful_uniform_full_int/RngReadAndSkip¨
-stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-stateful_uniform_full_int/strided_slice/stack¬
/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_1¬
/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice/stack_2
'stateful_uniform_full_int/strided_sliceStridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:06stateful_uniform_full_int/strided_slice/stack:output:08stateful_uniform_full_int/strided_slice/stack_1:output:08stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2)
'stateful_uniform_full_int/strided_slice´
!stateful_uniform_full_int/BitcastBitcast0stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02#
!stateful_uniform_full_int/Bitcast¬
/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/stateful_uniform_full_int/strided_slice_1/stack°
1stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_1°
1stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int/strided_slice_1/stack_2ü
)stateful_uniform_full_int/strided_slice_1StridedSlice0stateful_uniform_full_int/RngReadAndSkip:value:08stateful_uniform_full_int/strided_slice_1/stack:output:0:stateful_uniform_full_int/strided_slice_1/stack_1:output:0:stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2+
)stateful_uniform_full_int/strided_slice_1º
#stateful_uniform_full_int/Bitcast_1Bitcast2stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int/Bitcast_1
stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform_full_int/alg®
stateful_uniform_full_intStatelessRandomUniformFullIntV2(stateful_uniform_full_int/shape:output:0,stateful_uniform_full_int/Bitcast_1:output:0*stateful_uniform_full_int/Bitcast:output:0&stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_intb

zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2

zeros_like
stackPack"stateful_uniform_full_int:output:0zeros_like:output:0*
N*
T0	*
_output_shapes

:2
stack{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2
strided_sliceStridedSlicestack:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice
&stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:2(
&stateless_random_flip_left_right/Shape¿
4stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ26
4stateless_random_flip_left_right/strided_slice/stackº
6stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6stateless_random_flip_left_right/strided_slice/stack_1º
6stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6stateless_random_flip_left_right/strided_slice/stack_2¤
.stateless_random_flip_left_right/strided_sliceStridedSlice/stateless_random_flip_left_right/Shape:output:0=stateless_random_flip_left_right/strided_slice/stack:output:0?stateless_random_flip_left_right/strided_slice/stack_1:output:0?stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask20
.stateless_random_flip_left_right/strided_slice²
6stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 28
6stateless_random_flip_left_right/assert_positive/Const­
Astateless_random_flip_left_right/assert_positive/assert_less/LessLess?stateless_random_flip_left_right/assert_positive/Const:output:07stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2C
Astateless_random_flip_left_right/assert_positive/assert_less/LessÒ
Bstateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bstateless_random_flip_left_right/assert_positive/assert_less/Const·
@stateless_random_flip_left_right/assert_positive/assert_less/AllAllEstateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Kstateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2B
@stateless_random_flip_left_right/assert_positive/assert_less/All
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2K
Istateless_random_flip_left_right/assert_positive/assert_less/Assert/Const
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2S
Qstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0ë
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertIstateless_random_flip_left_right/assert_positive/assert_less/All:output:0Zstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2L
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert
%stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :2'
%stateless_random_flip_left_right/Rank´
7stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :29
7stateless_random_flip_left_right/assert_greater_equal/y«
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual.stateless_random_flip_left_right/Rank:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2D
Bstateless_random_flip_left_right/assert_greater_equal/GreaterEqualº
:stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2<
:stateless_random_flip_left_right/assert_greater_equal/RankÈ
Astateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2C
Astateless_random_flip_left_right/assert_greater_equal/range/startÈ
Astateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2C
Astateless_random_flip_left_right/assert_greater_equal/range/deltaú
;stateless_random_flip_left_right/assert_greater_equal/rangeRangeJstateless_random_flip_left_right/assert_greater_equal/range/start:output:0Cstateless_random_flip_left_right/assert_greater_equal/Rank:output:0Jstateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2=
;stateless_random_flip_left_right/assert_greater_equal/range£
9stateless_random_flip_left_right/assert_greater_equal/AllAllFstateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Dstateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2;
9stateless_random_flip_left_right/assert_greater_equal/Allô
Bstateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2D
Bstateless_random_flip_left_right/assert_greater_equal/Assert/Constø
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_1û
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_2
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2F
Dstateless_random_flip_left_right/assert_greater_equal/Assert/Const_3
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*?
value6B4 B.x (stateless_random_flip_left_right/Rank:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Q
valueHBF B@y (stateless_random_flip_left_right/assert_greater_equal/y:0) = 2L
Jstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertBstateless_random_flip_left_right/assert_greater_equal/All:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0.stateless_random_flip_left_right/Rank:output:0Sstateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0@stateless_random_flip_left_right/assert_greater_equal/y:output:0K^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2E
Cstateless_random_flip_left_right/assert_greater_equal/Assert/Assertã
3stateless_random_flip_left_right/control_dependencyIdentityinputsD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3stateless_random_flip_left_right/control_dependencyÅ
?stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2A
?stateless_random_flip_left_right/stateless_random_uniform/shapeÃ
=stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2?
=stateless_random_flip_left_right/stateless_random_uniform/minÃ
=stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2?
=stateless_random_flip_left_right/stateless_random_uniform/maxÐ
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice:output:0D^stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2X
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¬
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Q
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg½
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Hstateless_random_flip_left_right/stateless_random_uniform/shape:output:0\stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ustateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2T
Rstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2¶
=stateless_random_flip_left_right/stateless_random_uniform/subSubFstateless_random_flip_left_right/stateless_random_uniform/max:output:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/subÆ
=stateless_random_flip_left_right/stateless_random_uniform/mulMul[stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Astateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2?
=stateless_random_flip_left_right/stateless_random_uniform/mul«
9stateless_random_flip_left_right/stateless_random_uniformAddV2Astateless_random_flip_left_right/stateless_random_uniform/mul:z:0Fstateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2;
9stateless_random_flip_left_right/stateless_random_uniform
'stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'stateless_random_flip_left_right/Less/yè
%stateless_random_flip_left_right/LessLess=stateless_random_flip_left_right/stateless_random_uniform:z:00stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: 2'
%stateless_random_flip_left_right/Less
 stateless_random_flip_left_rightStatelessIf)stateless_random_flip_left_right/Less:z:0<stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *@
else_branch1R/
-stateless_random_flip_left_right_false_164423*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*?
then_branch0R.
,stateless_random_flip_left_right_true_1644222"
 stateless_random_flip_left_rightÄ
)stateless_random_flip_left_right/IdentityIdentity)stateless_random_flip_left_right:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateless_random_flip_left_right/Identity
!stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2#
!stateful_uniform_full_int_1/shape
!stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!stateful_uniform_full_int_1/ConstÅ
 stateful_uniform_full_int_1/ProdProd*stateful_uniform_full_int_1/shape:output:0*stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2"
 stateful_uniform_full_int_1/Prod
"stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2$
"stateful_uniform_full_int_1/Cast/x«
"stateful_uniform_full_int_1/Cast_1Cast)stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2$
"stateful_uniform_full_int_1/Cast_1¹
*stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip1stateful_uniform_full_int_rngreadandskip_resource+stateful_uniform_full_int_1/Cast/x:output:0&stateful_uniform_full_int_1/Cast_1:y:0)^stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:2,
*stateful_uniform_full_int_1/RngReadAndSkip¬
/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/stateful_uniform_full_int_1/strided_slice/stack°
1stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int_1/strided_slice/stack_1°
1stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int_1/strided_slice/stack_2
)stateful_uniform_full_int_1/strided_sliceStridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:08stateful_uniform_full_int_1/strided_slice/stack:output:0:stateful_uniform_full_int_1/strided_slice/stack_1:output:0:stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2+
)stateful_uniform_full_int_1/strided_sliceº
#stateful_uniform_full_int_1/BitcastBitcast2stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type02%
#stateful_uniform_full_int_1/Bitcast°
1stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1stateful_uniform_full_int_1/strided_slice_1/stack´
3stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3stateful_uniform_full_int_1/strided_slice_1/stack_1´
3stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3stateful_uniform_full_int_1/strided_slice_1/stack_2
+stateful_uniform_full_int_1/strided_slice_1StridedSlice2stateful_uniform_full_int_1/RngReadAndSkip:value:0:stateful_uniform_full_int_1/strided_slice_1/stack:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0<stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2-
+stateful_uniform_full_int_1/strided_slice_1À
%stateful_uniform_full_int_1/Bitcast_1Bitcast4stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02'
%stateful_uniform_full_int_1/Bitcast_1
stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2!
stateful_uniform_full_int_1/algº
stateful_uniform_full_int_1StatelessRandomUniformFullIntV2*stateful_uniform_full_int_1/shape:output:0.stateful_uniform_full_int_1/Bitcast_1:output:0,stateful_uniform_full_int_1/Bitcast:output:0(stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	2
stateful_uniform_full_int_1f
zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2
zeros_like_1
stack_1Pack$stateful_uniform_full_int_1:output:0zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2	
stack_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSlicestack_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1¬
#stateless_random_flip_up_down/ShapeShape2stateless_random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:2%
#stateless_random_flip_up_down/Shape¹
1stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ23
1stateless_random_flip_up_down/strided_slice/stack´
3stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 25
3stateless_random_flip_up_down/strided_slice/stack_1´
3stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3stateless_random_flip_up_down/strided_slice/stack_2
+stateless_random_flip_up_down/strided_sliceStridedSlice,stateless_random_flip_up_down/Shape:output:0:stateless_random_flip_up_down/strided_slice/stack:output:0<stateless_random_flip_up_down/strided_slice/stack_1:output:0<stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2-
+stateless_random_flip_up_down/strided_slice¬
3stateless_random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 25
3stateless_random_flip_up_down/assert_positive/Const¡
>stateless_random_flip_up_down/assert_positive/assert_less/LessLess<stateless_random_flip_up_down/assert_positive/Const:output:04stateless_random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:2@
>stateless_random_flip_up_down/assert_positive/assert_less/LessÌ
?stateless_random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2A
?stateless_random_flip_up_down/assert_positive/assert_less/Const«
=stateless_random_flip_up_down/assert_positive/assert_less/AllAllBstateless_random_flip_up_down/assert_positive/assert_less/Less:z:0Hstateless_random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2?
=stateless_random_flip_up_down/assert_positive/assert_less/Allû
Fstateless_random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2H
Fstateless_random_flip_up_down/assert_positive/assert_less/Assert/Const
Nstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2P
Nstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0±
Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertFstateless_random_flip_up_down/assert_positive/assert_less/All:output:0Wstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0P^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg*

T
2*
_output_shapes
 2I
Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert
"stateless_random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :2$
"stateless_random_flip_up_down/Rank®
4stateless_random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :26
4stateless_random_flip_up_down/assert_greater_equal/y
?stateless_random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual+stateless_random_flip_up_down/Rank:output:0=stateless_random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2A
?stateless_random_flip_up_down/assert_greater_equal/GreaterEqual´
7stateless_random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 29
7stateless_random_flip_up_down/assert_greater_equal/RankÂ
>stateless_random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2@
>stateless_random_flip_up_down/assert_greater_equal/range/startÂ
>stateless_random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2@
>stateless_random_flip_up_down/assert_greater_equal/range/deltaë
8stateless_random_flip_up_down/assert_greater_equal/rangeRangeGstateless_random_flip_up_down/assert_greater_equal/range/start:output:0@stateless_random_flip_up_down/assert_greater_equal/Rank:output:0Gstateless_random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2:
8stateless_random_flip_up_down/assert_greater_equal/range
6stateless_random_flip_up_down/assert_greater_equal/AllAllCstateless_random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Astateless_random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: 28
6stateless_random_flip_up_down/assert_greater_equal/Allî
?stateless_random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2A
?stateless_random_flip_up_down/assert_greater_equal/Assert/Constò
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2C
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_1ò
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (stateless_random_flip_up_down/Rank:0) = 2C
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_2
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (stateless_random_flip_up_down/assert_greater_equal/y:0) = 2C
Astateless_random_flip_up_down/assert_greater_equal/Assert/Const_3þ
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0þ
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1þ
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*<
value3B1 B+x (stateless_random_flip_up_down/Rank:0) = 2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*N
valueEBC B=y (stateless_random_flip_up_down/assert_greater_equal/y:0) = 2I
Gstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4ô
@stateless_random_flip_up_down/assert_greater_equal/Assert/AssertAssert?stateless_random_flip_up_down/assert_greater_equal/All:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:0+stateless_random_flip_up_down/Rank:output:0Pstateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0=stateless_random_flip_up_down/assert_greater_equal/y:output:0H^stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2B
@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert¦
0stateless_random_flip_up_down/control_dependencyIdentity2stateless_random_flip_left_right/Identity:output:0A^stateless_random_flip_up_down/assert_greater_equal/Assert/AssertH^stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*<
_class2
0.loc:@stateless_random_flip_left_right/Identity*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0stateless_random_flip_up_down/control_dependency¿
<stateless_random_flip_up_down/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2>
<stateless_random_flip_up_down/stateless_random_uniform/shape½
:stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:stateless_random_flip_up_down/stateless_random_uniform/min½
:stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2<
:stateless_random_flip_up_down/stateless_random_uniform/maxÉ
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterstrided_slice_1:output:0A^stateless_random_flip_up_down/assert_greater_equal/Assert/Assert* 
_output_shapes
::2U
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter£
Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgT^stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2N
Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg«
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Estateless_random_flip_up_down/stateless_random_uniform/shape:output:0Ystateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0]stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Rstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2Q
Ostateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2ª
:stateless_random_flip_up_down/stateless_random_uniform/subSubCstateless_random_flip_up_down/stateless_random_uniform/max:output:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2<
:stateless_random_flip_up_down/stateless_random_uniform/subº
:stateless_random_flip_up_down/stateless_random_uniform/mulMulXstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0>stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2<
:stateless_random_flip_up_down/stateless_random_uniform/mul
6stateless_random_flip_up_down/stateless_random_uniformAddV2>stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Cstateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 28
6stateless_random_flip_up_down/stateless_random_uniform
$stateless_random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$stateless_random_flip_up_down/Less/yÜ
"stateless_random_flip_up_down/LessLess:stateless_random_flip_up_down/stateless_random_uniform:z:0-stateless_random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: 2$
"stateless_random_flip_up_down/Lessõ
stateless_random_flip_up_downStatelessIf&stateless_random_flip_up_down/Less:z:09stateless_random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *=
else_branch.R,
*stateless_random_flip_up_down_false_164499*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
then_branch-R+
)stateless_random_flip_up_down_true_1644982
stateless_random_flip_up_down»
&stateless_random_flip_up_down/IdentityIdentity&stateless_random_flip_up_down:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_up_down/Identity
IdentityIdentity/stateless_random_flip_up_down/Identity:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp)^stateful_uniform_full_int/RngReadAndSkip+^stateful_uniform_full_int_1/RngReadAndSkipD^stateless_random_flip_left_right/assert_greater_equal/Assert/AssertK^stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertP^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgW^stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterA^stateless_random_flip_up_down/assert_greater_equal/Assert/AssertH^stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertM^stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgT^stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2T
(stateful_uniform_full_int/RngReadAndSkip(stateful_uniform_full_int/RngReadAndSkip2X
*stateful_uniform_full_int_1/RngReadAndSkip*stateful_uniform_full_int_1/RngReadAndSkip2
Cstateless_random_flip_left_right/assert_greater_equal/Assert/AssertCstateless_random_flip_left_right/assert_greater_equal/Assert/Assert2
Jstateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertJstateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2¢
Ostateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgOstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2°
Vstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterVstateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert@stateless_random_flip_up_down/assert_greater_equal/Assert/Assert2
Gstateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertGstateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert2
Lstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgLstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2ª
Sstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterSstateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
Á
-stateless_random_flip_left_right_false_164423a
]stateless_random_flip_left_right_identity_stateless_random_flip_left_right_control_dependency-
)stateless_random_flip_left_right_identityø
)stateless_random_flip_left_right/IdentityIdentity]stateless_random_flip_left_right_identity_stateless_random_flip_left_right_control_dependency*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateless_random_flip_left_right/Identity"_
)stateless_random_flip_left_right_identity2stateless_random_flip_left_right/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

õ
D__inference_dense_31_layer_call_and_return_conditional_losses_161439

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ßg
ë
I__inference_sequential_17_layer_call_and_return_conditional_losses_162786
sequential_inputB
(conv2d_90_conv2d_readvariableop_resource: 7
)conv2d_90_biasadd_readvariableop_resource: B
(conv2d_91_conv2d_readvariableop_resource: @7
)conv2d_91_biasadd_readvariableop_resource:@B
(conv2d_92_conv2d_readvariableop_resource:@@7
)conv2d_92_biasadd_readvariableop_resource:@B
(conv2d_93_conv2d_readvariableop_resource:@@7
)conv2d_93_biasadd_readvariableop_resource:@B
(conv2d_94_conv2d_readvariableop_resource:@@7
)conv2d_94_biasadd_readvariableop_resource:@B
(conv2d_95_conv2d_readvariableop_resource:@@7
)conv2d_95_biasadd_readvariableop_resource:@:
'dense_30_matmul_readvariableop_resource:	@6
(dense_30_biasadd_readvariableop_resource:@9
'dense_31_matmul_readvariableop_resource:@6
(dense_31_biasadd_readvariableop_resource:
identity¢ conv2d_90/BiasAdd/ReadVariableOp¢conv2d_90/Conv2D/ReadVariableOp¢ conv2d_91/BiasAdd/ReadVariableOp¢conv2d_91/Conv2D/ReadVariableOp¢ conv2d_92/BiasAdd/ReadVariableOp¢conv2d_92/Conv2D/ReadVariableOp¢ conv2d_93/BiasAdd/ReadVariableOp¢conv2d_93/Conv2D/ReadVariableOp¢ conv2d_94/BiasAdd/ReadVariableOp¢conv2d_94/Conv2D/ReadVariableOp¢ conv2d_95/BiasAdd/ReadVariableOp¢conv2d_95/Conv2D/ReadVariableOp¢dense_30/BiasAdd/ReadVariableOp¢dense_30/MatMul/ReadVariableOp¢dense_31/BiasAdd/ReadVariableOp¢dense_31/MatMul/ReadVariableOp
sequential/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
sequential/rescaling_1/Cast/x
sequential/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
sequential/rescaling_1/Cast_1/xµ
sequential/rescaling_1/mulMulsequential_input&sequential/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/mulÇ
sequential/rescaling_1/addAddV2sequential/rescaling_1/mul:z:0(sequential/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
sequential/rescaling_1/add
sequential/resizing/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2!
sequential/resizing/resize/size
)sequential/resizing/resize/ResizeBilinearResizeBilinearsequential/rescaling_1/add:z:0(sequential/resizing/resize/size:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
half_pixel_centers(2+
)sequential/resizing/resize/ResizeBilinear³
conv2d_90/Conv2D/ReadVariableOpReadVariableOp(conv2d_90_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_90/Conv2D/ReadVariableOpø
conv2d_90/Conv2DConv2D:sequential/resizing/resize/ResizeBilinear:resized_images:0'conv2d_90/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2
conv2d_90/Conv2Dª
 conv2d_90/BiasAdd/ReadVariableOpReadVariableOp)conv2d_90_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_90/BiasAdd/ReadVariableOp²
conv2d_90/BiasAddBiasAddconv2d_90/Conv2D:output:0(conv2d_90/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/BiasAdd
conv2d_90/ReluReluconv2d_90/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
conv2d_90/ReluÊ
max_pooling2d_90/MaxPoolMaxPoolconv2d_90/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_90/MaxPool³
conv2d_91/Conv2D/ReadVariableOpReadVariableOp(conv2d_91_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_91/Conv2D/ReadVariableOpÝ
conv2d_91/Conv2DConv2D!max_pooling2d_90/MaxPool:output:0'conv2d_91/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2
conv2d_91/Conv2Dª
 conv2d_91/BiasAdd/ReadVariableOpReadVariableOp)conv2d_91_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_91/BiasAdd/ReadVariableOp°
conv2d_91/BiasAddBiasAddconv2d_91/Conv2D:output:0(conv2d_91/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/BiasAdd~
conv2d_91/ReluReluconv2d_91/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
conv2d_91/ReluÊ
max_pooling2d_91/MaxPoolMaxPoolconv2d_91/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_91/MaxPool³
conv2d_92/Conv2D/ReadVariableOpReadVariableOp(conv2d_92_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_92/Conv2D/ReadVariableOpÝ
conv2d_92/Conv2DConv2D!max_pooling2d_91/MaxPool:output:0'conv2d_92/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*
paddingVALID*
strides
2
conv2d_92/Conv2Dª
 conv2d_92/BiasAdd/ReadVariableOpReadVariableOp)conv2d_92_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_92/BiasAdd/ReadVariableOp°
conv2d_92/BiasAddBiasAddconv2d_92/Conv2D:output:0(conv2d_92/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/BiasAdd~
conv2d_92/ReluReluconv2d_92/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@2
conv2d_92/ReluÊ
max_pooling2d_92/MaxPoolMaxPoolconv2d_92/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_92/MaxPool³
conv2d_93/Conv2D/ReadVariableOpReadVariableOp(conv2d_93_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_93/Conv2D/ReadVariableOpÝ
conv2d_93/Conv2DConv2D!max_pooling2d_92/MaxPool:output:0'conv2d_93/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_93/Conv2Dª
 conv2d_93/BiasAdd/ReadVariableOpReadVariableOp)conv2d_93_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_93/BiasAdd/ReadVariableOp°
conv2d_93/BiasAddBiasAddconv2d_93/Conv2D:output:0(conv2d_93/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/BiasAdd~
conv2d_93/ReluReluconv2d_93/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_93/ReluÊ
max_pooling2d_93/MaxPoolMaxPoolconv2d_93/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_93/MaxPool³
conv2d_94/Conv2D/ReadVariableOpReadVariableOp(conv2d_94_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_94/Conv2D/ReadVariableOpÝ
conv2d_94/Conv2DConv2D!max_pooling2d_93/MaxPool:output:0'conv2d_94/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_94/Conv2Dª
 conv2d_94/BiasAdd/ReadVariableOpReadVariableOp)conv2d_94_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_94/BiasAdd/ReadVariableOp°
conv2d_94/BiasAddBiasAddconv2d_94/Conv2D:output:0(conv2d_94/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/BiasAdd~
conv2d_94/ReluReluconv2d_94/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_94/ReluÊ
max_pooling2d_94/MaxPoolMaxPoolconv2d_94/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_94/MaxPool³
conv2d_95/Conv2D/ReadVariableOpReadVariableOp(conv2d_95_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_95/Conv2D/ReadVariableOpÝ
conv2d_95/Conv2DConv2D!max_pooling2d_94/MaxPool:output:0'conv2d_95/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
conv2d_95/Conv2Dª
 conv2d_95/BiasAdd/ReadVariableOpReadVariableOp)conv2d_95_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_95/BiasAdd/ReadVariableOp°
conv2d_95/BiasAddBiasAddconv2d_95/Conv2D:output:0(conv2d_95/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/BiasAdd~
conv2d_95/ReluReluconv2d_95/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_95/ReluÊ
max_pooling2d_95/MaxPoolMaxPoolconv2d_95/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
max_pooling2d_95/MaxPoolu
flatten_15/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
flatten_15/Const¤
flatten_15/ReshapeReshape!max_pooling2d_95/MaxPool:output:0flatten_15/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten_15/Reshape©
dense_30/MatMul/ReadVariableOpReadVariableOp'dense_30_matmul_readvariableop_resource*
_output_shapes
:	@*
dtype02 
dense_30/MatMul/ReadVariableOp£
dense_30/MatMulMatMulflatten_15/Reshape:output:0&dense_30/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/MatMul§
dense_30/BiasAdd/ReadVariableOpReadVariableOp(dense_30_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
dense_30/BiasAdd/ReadVariableOp¥
dense_30/BiasAddBiasAdddense_30/MatMul:product:0'dense_30/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/BiasAdds
dense_30/ReluReludense_30/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dense_30/Relu¨
dense_31/MatMul/ReadVariableOpReadVariableOp'dense_31_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02 
dense_31/MatMul/ReadVariableOp£
dense_31/MatMulMatMuldense_30/Relu:activations:0&dense_31/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/MatMul§
dense_31/BiasAdd/ReadVariableOpReadVariableOp(dense_31_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_31/BiasAdd/ReadVariableOp¥
dense_31/BiasAddBiasAdddense_31/MatMul:product:0'dense_31/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/BiasAdd|
dense_31/SoftmaxSoftmaxdense_31/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_31/Softmaxu
IdentityIdentitydense_31/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityò
NoOpNoOp!^conv2d_90/BiasAdd/ReadVariableOp ^conv2d_90/Conv2D/ReadVariableOp!^conv2d_91/BiasAdd/ReadVariableOp ^conv2d_91/Conv2D/ReadVariableOp!^conv2d_92/BiasAdd/ReadVariableOp ^conv2d_92/Conv2D/ReadVariableOp!^conv2d_93/BiasAdd/ReadVariableOp ^conv2d_93/Conv2D/ReadVariableOp!^conv2d_94/BiasAdd/ReadVariableOp ^conv2d_94/Conv2D/ReadVariableOp!^conv2d_95/BiasAdd/ReadVariableOp ^conv2d_95/Conv2D/ReadVariableOp ^dense_30/BiasAdd/ReadVariableOp^dense_30/MatMul/ReadVariableOp ^dense_31/BiasAdd/ReadVariableOp^dense_31/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2D
 conv2d_90/BiasAdd/ReadVariableOp conv2d_90/BiasAdd/ReadVariableOp2B
conv2d_90/Conv2D/ReadVariableOpconv2d_90/Conv2D/ReadVariableOp2D
 conv2d_91/BiasAdd/ReadVariableOp conv2d_91/BiasAdd/ReadVariableOp2B
conv2d_91/Conv2D/ReadVariableOpconv2d_91/Conv2D/ReadVariableOp2D
 conv2d_92/BiasAdd/ReadVariableOp conv2d_92/BiasAdd/ReadVariableOp2B
conv2d_92/Conv2D/ReadVariableOpconv2d_92/Conv2D/ReadVariableOp2D
 conv2d_93/BiasAdd/ReadVariableOp conv2d_93/BiasAdd/ReadVariableOp2B
conv2d_93/Conv2D/ReadVariableOpconv2d_93/Conv2D/ReadVariableOp2D
 conv2d_94/BiasAdd/ReadVariableOp conv2d_94/BiasAdd/ReadVariableOp2B
conv2d_94/Conv2D/ReadVariableOpconv2d_94/Conv2D/ReadVariableOp2D
 conv2d_95/BiasAdd/ReadVariableOp conv2d_95/BiasAdd/ReadVariableOp2B
conv2d_95/Conv2D/ReadVariableOpconv2d_95/Conv2D/ReadVariableOp2B
dense_30/BiasAdd/ReadVariableOpdense_30/BiasAdd/ReadVariableOp2@
dense_30/MatMul/ReadVariableOpdense_30/MatMul/ReadVariableOp2B
dense_31/BiasAdd/ReadVariableOpdense_31/BiasAdd/ReadVariableOp2@
dense_31/MatMul/ReadVariableOpdense_31/MatMul/ReadVariableOp:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namesequential_input
à
Ä
K__inference_random_rotation_layer_call_and_return_conditional_losses_160886

inputs6
(stateful_uniform_rngreadandskip_resource:	
identity¢stateful_uniform/RngReadAndSkipb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimsQ
ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ì
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1^
CastCaststrided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ì
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2b
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
Cast_1~
stateful_uniform/shapePackstrided_slice:output:0*
N*
T0*
_output_shapes
:2
stateful_uniform/shapeq
stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2
stateful_uniform/minq
stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2
stateful_uniform/maxz
stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
stateful_uniform/Const
stateful_uniform/ProdProdstateful_uniform/shape:output:0stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2
stateful_uniform/Prodt
stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2
stateful_uniform/Cast/x
stateful_uniform/Cast_1Caststateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
stateful_uniform/Cast_1Ù
stateful_uniform/RngReadAndSkipRngReadAndSkip(stateful_uniform_rngreadandskip_resource stateful_uniform/Cast/x:output:0stateful_uniform/Cast_1:y:0*
_output_shapes
:2!
stateful_uniform/RngReadAndSkip
$stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$stateful_uniform/strided_slice/stack
&stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_1
&stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice/stack_2Î
stateful_uniform/strided_sliceStridedSlice'stateful_uniform/RngReadAndSkip:value:0-stateful_uniform/strided_slice/stack:output:0/stateful_uniform/strided_slice/stack_1:output:0/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2 
stateful_uniform/strided_slice
stateful_uniform/BitcastBitcast'stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast
&stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&stateful_uniform/strided_slice_1/stack
(stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_1
(stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(stateful_uniform/strided_slice_1/stack_2Æ
 stateful_uniform/strided_slice_1StridedSlice'stateful_uniform/RngReadAndSkip:value:0/stateful_uniform/strided_slice_1/stack:output:01stateful_uniform/strided_slice_1/stack_1:output:01stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2"
 stateful_uniform/strided_slice_1
stateful_uniform/Bitcast_1Bitcast)stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02
stateful_uniform/Bitcast_1 
-stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2/
-stateful_uniform/StatelessRandomUniformV2/alg¸
)stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2stateful_uniform/shape:output:0#stateful_uniform/Bitcast_1:output:0!stateful_uniform/Bitcast:output:06stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)stateful_uniform/StatelessRandomUniformV2
stateful_uniform/subSubstateful_uniform/max:output:0stateful_uniform/min:output:0*
T0*
_output_shapes
: 2
stateful_uniform/sub¯
stateful_uniform/mulMul2stateful_uniform/StatelessRandomUniformV2:output:0stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniform/mul
stateful_uniformAddV2stateful_uniform/mul:z:0stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
stateful_uniforms
rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub/y~
rotation_matrix/subSub
Cast_1:y:0rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/subu
rotation_matrix/CosCosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cosw
rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_1/y
rotation_matrix/sub_1Sub
Cast_1:y:0 rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_1
rotation_matrix/mulMulrotation_matrix/Cos:y:0rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mulu
rotation_matrix/SinSinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sinw
rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_2/y
rotation_matrix/sub_2SubCast:y:0 rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_2
rotation_matrix/mul_1Mulrotation_matrix/Sin:y:0rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_1
rotation_matrix/sub_3Subrotation_matrix/mul:z:0rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_3
rotation_matrix/sub_4Subrotation_matrix/sub:z:0rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_4{
rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv/yª
rotation_matrix/truedivRealDivrotation_matrix/sub_4:z:0"rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truedivw
rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_5/y
rotation_matrix/sub_5SubCast:y:0 rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_5y
rotation_matrix/Sin_1Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_1w
rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_6/y
rotation_matrix/sub_6Sub
Cast_1:y:0 rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_6
rotation_matrix/mul_2Mulrotation_matrix/Sin_1:y:0rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_2y
rotation_matrix/Cos_1Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_1w
rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
rotation_matrix/sub_7/y
rotation_matrix/sub_7SubCast:y:0 rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2
rotation_matrix/sub_7
rotation_matrix/mul_3Mulrotation_matrix/Cos_1:y:0rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/mul_3
rotation_matrix/addAddV2rotation_matrix/mul_2:z:0rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/add
rotation_matrix/sub_8Subrotation_matrix/sub_5:z:0rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/sub_8
rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
rotation_matrix/truediv_1/y°
rotation_matrix/truediv_1RealDivrotation_matrix/sub_8:z:0$rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/truediv_1r
rotation_matrix/ShapeShapestateful_uniform:z:0*
T0*
_output_shapes
:2
rotation_matrix/Shape
#rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#rotation_matrix/strided_slice/stack
%rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_1
%rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%rotation_matrix/strided_slice/stack_2Â
rotation_matrix/strided_sliceStridedSlicerotation_matrix/Shape:output:0,rotation_matrix/strided_slice/stack:output:0.rotation_matrix/strided_slice/stack_1:output:0.rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
rotation_matrix/strided_slicey
rotation_matrix/Cos_2Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_2
%rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_1/stack£
'rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_1/stack_1£
'rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_1/stack_2÷
rotation_matrix/strided_slice_1StridedSlicerotation_matrix/Cos_2:y:0.rotation_matrix/strided_slice_1/stack:output:00rotation_matrix/strided_slice_1/stack_1:output:00rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_1y
rotation_matrix/Sin_2Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_2
%rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_2/stack£
'rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_2/stack_1£
'rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_2/stack_2÷
rotation_matrix/strided_slice_2StridedSlicerotation_matrix/Sin_2:y:0.rotation_matrix/strided_slice_2/stack:output:00rotation_matrix/strided_slice_2/stack_1:output:00rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_2
rotation_matrix/NegNeg(rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Neg
%rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_3/stack£
'rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_3/stack_1£
'rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_3/stack_2ù
rotation_matrix/strided_slice_3StridedSlicerotation_matrix/truediv:z:0.rotation_matrix/strided_slice_3/stack:output:00rotation_matrix/strided_slice_3/stack_1:output:00rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_3y
rotation_matrix/Sin_3Sinstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Sin_3
%rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_4/stack£
'rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_4/stack_1£
'rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_4/stack_2÷
rotation_matrix/strided_slice_4StridedSlicerotation_matrix/Sin_3:y:0.rotation_matrix/strided_slice_4/stack:output:00rotation_matrix/strided_slice_4/stack_1:output:00rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_4y
rotation_matrix/Cos_3Cosstateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/Cos_3
%rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_5/stack£
'rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_5/stack_1£
'rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_5/stack_2÷
rotation_matrix/strided_slice_5StridedSlicerotation_matrix/Cos_3:y:0.rotation_matrix/strided_slice_5/stack:output:00rotation_matrix/strided_slice_5/stack_1:output:00rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_5
%rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%rotation_matrix/strided_slice_6/stack£
'rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'rotation_matrix/strided_slice_6/stack_1£
'rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'rotation_matrix/strided_slice_6/stack_2û
rotation_matrix/strided_slice_6StridedSlicerotation_matrix/truediv_1:z:0.rotation_matrix/strided_slice_6/stack:output:00rotation_matrix/strided_slice_6/stack_1:output:00rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask2!
rotation_matrix/strided_slice_6
rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2 
rotation_matrix/zeros/packed/1Ã
rotation_matrix/zeros/packedPack&rotation_matrix/strided_slice:output:0'rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
rotation_matrix/zeros/packed
rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rotation_matrix/zeros/Constµ
rotation_matrix/zerosFill%rotation_matrix/zeros/packed:output:0$rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/zeros|
rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
rotation_matrix/concat/axis¨
rotation_matrix/concatConcatV2(rotation_matrix/strided_slice_1:output:0rotation_matrix/Neg:y:0(rotation_matrix/strided_slice_3:output:0(rotation_matrix/strided_slice_4:output:0(rotation_matrix/strided_slice_5:output:0(rotation_matrix/strided_slice_6:output:0rotation_matrix/zeros:output:0$rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
rotation_matrix/concate
transform/ShapeShapeExpandDims:output:0*
T0*
_output_shapes
:2
transform/Shape
transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
transform/strided_slice/stack
transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_1
transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
transform/strided_slice/stack_2
transform/strided_sliceStridedSlicetransform/Shape:output:0&transform/strided_slice/stack:output:0(transform/strided_slice/stack_1:output:0(transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
transform/strided_sliceq
transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2
transform/fill_valueÝ
$transform/ImageProjectiveTransformV3ImageProjectiveTransformV3ExpandDims:output:0rotation_matrix/concat:output:0 transform/strided_slice:output:0transform/fill_value:output:0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2&
$transform/ImageProjectiveTransformV3¦
SqueezeSqueeze9transform/ImageProjectiveTransformV3:transformed_images:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2	
Squeezep
IdentityIdentitySqueeze:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityp
NoOpNoOp ^stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2B
stateful_uniform/RngReadAndSkipstateful_uniform/RngReadAndSkip:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_164175

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_164260

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨J
Ó
I__inference_sequential_17_layer_call_and_return_conditional_losses_161446

inputs*
conv2d_90_161277: 
conv2d_90_161279: *
conv2d_91_161300: @
conv2d_91_161302:@*
conv2d_92_161323:@@
conv2d_92_161325:@*
conv2d_93_161346:@@
conv2d_93_161348:@*
conv2d_94_161369:@@
conv2d_94_161371:@*
conv2d_95_161392:@@
conv2d_95_161394:@"
dense_30_161423:	@
dense_30_161425:@!
dense_31_161440:@
dense_31_161442:
identity¢!conv2d_90/StatefulPartitionedCall¢!conv2d_91/StatefulPartitionedCall¢!conv2d_92/StatefulPartitionedCall¢!conv2d_93/StatefulPartitionedCall¢!conv2d_94/StatefulPartitionedCall¢!conv2d_95/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCallä
sequential/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1612582
sequential/PartitionedCall
sequential_1/PartitionedCallPartitionedCall#sequential/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1612632
sequential_1/PartitionedCallÂ
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCall%sequential_1/PartitionedCall:output:0conv2d_90_161277conv2d_90_161279*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_90_layer_call_and_return_conditional_losses_1612762#
!conv2d_90/StatefulPartitionedCall
 max_pooling2d_90/PartitionedCallPartitionedCall*conv2d_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_1612862"
 max_pooling2d_90/PartitionedCallÄ
!conv2d_91/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_90/PartitionedCall:output:0conv2d_91_161300conv2d_91_161302*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_91_layer_call_and_return_conditional_losses_1612992#
!conv2d_91/StatefulPartitionedCall
 max_pooling2d_91/PartitionedCallPartitionedCall*conv2d_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_1613092"
 max_pooling2d_91/PartitionedCallÄ
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_91/PartitionedCall:output:0conv2d_92_161323conv2d_92_161325*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1613222#
!conv2d_92/StatefulPartitionedCall
 max_pooling2d_92/PartitionedCallPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_1613322"
 max_pooling2d_92/PartitionedCallÄ
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_92/PartitionedCall:output:0conv2d_93_161346conv2d_93_161348*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1613452#
!conv2d_93/StatefulPartitionedCall
 max_pooling2d_93/PartitionedCallPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_1613552"
 max_pooling2d_93/PartitionedCallÄ
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_93/PartitionedCall:output:0conv2d_94_161369conv2d_94_161371*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1613682#
!conv2d_94/StatefulPartitionedCall
 max_pooling2d_94/PartitionedCallPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_1613782"
 max_pooling2d_94/PartitionedCallÄ
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_94/PartitionedCall:output:0conv2d_95_161392conv2d_95_161394*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1613912#
!conv2d_95/StatefulPartitionedCall
 max_pooling2d_95/PartitionedCallPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_1614012"
 max_pooling2d_95/PartitionedCallþ
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_95/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1614092
flatten_15/PartitionedCall±
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_30_161423dense_30_161425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_1614222"
 dense_30/StatefulPartitionedCall·
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_161440dense_31_161442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_1614392"
 dense_31/StatefulPartitionedCall
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityì
NoOpNoOp"^conv2d_90/StatefulPartitionedCall"^conv2d_91/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall2F
!conv2d_91/StatefulPartitionedCall!conv2d_91/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å

0__inference_random_rotation_layer_call_fn_164523

inputs
unknown:	
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_random_rotation_layer_call_and_return_conditional_losses_1608862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ö
D__inference_dense_30_layer_call_and_return_conditional_losses_161422

inputs1
matmul_readvariableop_resource:	@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@*
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
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
M
1__inference_max_pooling2d_91_layer_call_fn_164090

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_1613092
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ}}@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@
 
_user_specified_nameinputs
¿
h
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_164180

inputs
identity
MaxPoolMaxPoolinputs*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ä
G
+__inference_sequential_layer_call_fn_163096

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1607032
PartitionedCalli
IdentityIdentityPartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs


*__inference_conv2d_93_layer_call_fn_164149

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1613452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ù	
å
5random_flip_stateless_random_flip_up_down_true_163661t
prandom_flip_stateless_random_flip_up_down_reversev2_random_flip_stateless_random_flip_up_down_control_dependency6
2random_flip_stateless_random_flip_up_down_identity¾
8random_flip/stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip/stateless_random_flip_up_down/ReverseV2/axisã
3random_flip/stateless_random_flip_up_down/ReverseV2	ReverseV2prandom_flip_stateless_random_flip_up_down_reversev2_random_flip_stateless_random_flip_up_down_control_dependencyArandom_flip/stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3random_flip/stateless_random_flip_up_down/ReverseV2é
2random_flip/stateless_random_flip_up_down/IdentityIdentity<random_flip/stateless_random_flip_up_down/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_up_down/Identity"q
2random_flip_stateless_random_flip_up_down_identity;random_flip/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

c
G__inference_rescaling_1_layer_call_and_return_conditional_losses_160660

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/xa
mulMulinputsCast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
mulf
addAddV2mul:z:0Cast_1/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
´
Õ
.__inference_sequential_17_layer_call_fn_162224
sequential_input!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@#
	unknown_5:@@
	unknown_6:@#
	unknown_7:@@
	unknown_8:@#
	unknown_9:@@

unknown_10:@

unknown_11:	@

unknown_12:@

unknown_13:@

unknown_14:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallsequential_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_17_layer_call_and_return_conditional_losses_1614462
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:c _
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
_user_specified_namesequential_input
¹
µ
)stateless_random_flip_up_down_true_164498\
Xstateless_random_flip_up_down_reversev2_stateless_random_flip_up_down_control_dependency*
&stateless_random_flip_up_down_identity¦
,stateless_random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: 2.
,stateless_random_flip_up_down/ReverseV2/axis§
'stateless_random_flip_up_down/ReverseV2	ReverseV2Xstateless_random_flip_up_down_reversev2_stateless_random_flip_up_down_control_dependency5stateless_random_flip_up_down/ReverseV2/axis:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'stateless_random_flip_up_down/ReverseV2Å
&stateless_random_flip_up_down/IdentityIdentity0stateless_random_flip_up_down/ReverseV2:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&stateless_random_flip_up_down/Identity"Y
&stateless_random_flip_up_down_identity/stateless_random_flip_up_down/Identity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2 .
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
§
¡
-__inference_sequential_1_layer_call_fn_163215
random_flip_input
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1610782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input

c
G__inference_rescaling_1_layer_call_and_return_conditional_losses_164324

inputs
identityU
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *;2
Cast/xY
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2

Cast_1/xa
mulMulinputsCast/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
mulf
addAddV2mul:z:0Cast_1/x:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
add`
IdentityIdentityadd:z:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_161119

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
h
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_164055

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

õ
D__inference_dense_31_layer_call_and_return_conditional_losses_164311

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
ì
M
1__inference_max_pooling2d_92_layer_call_fn_164130

inputs
identityÒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_1613322
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ<<@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@
 
_user_specified_nameinputs
ï
`
D__inference_resizing_layer_call_and_return_conditional_losses_164338

inputs
identityp
resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
resize/ExpandDims/dim
resize/ExpandDims
ExpandDimsinputsresize/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
resize/ExpandDimsk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resize/size¿
resize/ResizeBilinearResizeBilinearresize/ExpandDims:output:0resize/size:output:0*
T0*(
_output_shapes
:*
half_pixel_centers(2
resize/ResizeBilinear
resize/SqueezeSqueeze&resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 2
resize/Squeezeh
IdentityIdentityresize/Squeeze:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_95_layer_call_and_return_conditional_losses_161391

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¤

*__inference_conv2d_90_layer_call_fn_164029

inputs!
unknown: 
	unknown_0: 
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_90_layer_call_and_return_conditional_losses_1612762
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ø
þ
E__inference_conv2d_90_layer_call_and_return_conditional_losses_164040

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¦
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2
Reluw
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï
`
D__inference_resizing_layer_call_and_return_conditional_losses_160671

inputs
identityp
resize/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
resize/ExpandDims/dim
resize/ExpandDims
ExpandDimsinputsresize/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
resize/ExpandDimsk
resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resize/size¿
resize/ResizeBilinearResizeBilinearresize/ExpandDims:output:0resize/size:output:0*
T0*(
_output_shapes
:*
half_pixel_centers(2
resize/ResizeBilinear
resize/SqueezeSqueeze&resize/ResizeBilinear:resized_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 2
resize/Squeezeh
IdentityIdentityresize/Squeeze:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs
Ù
M
1__inference_max_pooling2d_91_layer_call_fn_164085

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_1611412
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

c
G__inference_random_flip_layer_call_and_return_conditional_losses_160738

inputs
identity_
IdentityIdentityinputs*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á	
b
F__inference_sequential_layer_call_and_return_conditional_losses_160674

inputs
identityb
CastCastinputs*

DstT0*

SrcT0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ2
Castä
rescaling_1/PartitionedCallPartitionedCallCast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_rescaling_1_layer_call_and_return_conditional_losses_1606602
rescaling_1/PartitionedCallï
resizing/PartitionedCallPartitionedCall$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *$
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_resizing_layer_call_and_return_conditional_losses_1606712
resizing/PartitionedCallr
IdentityIdentity!resizing/PartitionedCall:output:0*
T0*$
_output_shapes
:2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿØ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
 
_user_specified_nameinputs


*__inference_conv2d_94_layer_call_fn_164189

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1613682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

g
K__inference_random_rotation_layer_call_and_return_conditional_losses_160749

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDims
SqueezeSqueezeExpandDims:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2	
Squeezei
IdentityIdentitySqueeze:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
ù
H__inference_sequential_1_layer_call_and_return_conditional_losses_163791
random_flip_inputK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	
identity¢4random_flip/stateful_uniform_full_int/RngReadAndSkip¢6random_flip/stateful_uniform_full_int_1/RngReadAndSkip¢Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert¢Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¢[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert¢Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert¢Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter¢/random_rotation/stateful_uniform/RngReadAndSkip¤
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_flip/stateful_uniform_full_int/shape¤
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+random_flip/stateful_uniform_full_int/Constí
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2,
*random_flip/stateful_uniform_full_int/Prod
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2.
,random_flip/stateful_uniform_full_int/Cast/xÉ
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int/Cast_1Â
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:26
4random_flip/stateful_uniform_full_int/RngReadAndSkipÀ
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip/stateful_uniform_full_int/strided_slice/stackÄ
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Ä
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Ì
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask25
3random_flip/stateful_uniform_full_int/strided_sliceØ
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02/
-random_flip/stateful_uniform_full_int/BitcastÄ
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice_1/stackÈ
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1È
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Ä
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:27
5random_flip/stateful_uniform_full_int/strided_slice_1Þ
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int/Bitcast_1
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_flip/stateful_uniform_full_int/algö
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2'
%random_flip/stateful_uniform_full_intz
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like±
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
random_flip/strided_slice/stack
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!random_flip/strided_slice/stack_1
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!random_flip/strided_slice/stack_2Ð
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_slice©
2random_flip/stateless_random_flip_left_right/ShapeShaperandom_flip_input*
T0*
_output_shapes
:24
2random_flip/stateless_random_flip_left_right/Shape×
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2B
@random_flip/stateless_random_flip_left_right/strided_slice/stackÒ
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Ò
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2ì
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2<
:random_flip/stateless_random_flip_left_right/strided_sliceÊ
Brandom_flip/stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2D
Brandom_flip/stateless_random_flip_left_right/assert_positive/ConstÝ
Mrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/LessLessKrandom_flip/stateless_random_flip_left_right/assert_positive/Const:output:0Crandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2O
Mrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Lessê
Nrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Constç
Lrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/AllAllQrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Wrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2N
Lrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/All
Urandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2W
Urandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Const©
]random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2_
]random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0
Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertUrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/All:output:0frandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2X
Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¨
1random_flip/stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :23
1random_flip/stateless_random_flip_left_right/RankÌ
Crandom_flip/stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2E
Crandom_flip/stateless_random_flip_left_right/assert_greater_equal/yÛ
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual:random_flip/stateless_random_flip_left_right/Rank:output:0Lrandom_flip/stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2P
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/GreaterEqualÒ
Frandom_flip/stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2H
Frandom_flip/stateless_random_flip_left_right/assert_greater_equal/Rankà
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/startà
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2O
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/delta¶
Grandom_flip/stateless_random_flip_left_right/assert_greater_equal/rangeRangeVrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/start:output:0Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Rank:output:0Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2I
Grandom_flip/stateless_random_flip_left_right/assert_greater_equal/rangeÓ
Erandom_flip/stateless_random_flip_left_right/assert_greater_equal/AllAllRrandom_flip/stateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2G
Erandom_flip/stateless_random_flip_left_right/assert_greater_equal/All
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2P
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2R
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_1
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (random_flip/stateless_random_flip_left_right/Rank:0) = 2R
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_2±
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*]
valueTBR BLy (random_flip/stateless_random_flip_left_right/assert_greater_equal/y:0) = 2R
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_3
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1«
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (random_flip/stateless_random_flip_left_right/Rank:0) = 2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2½
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*]
valueTBR BLy (random_flip/stateless_random_flip_left_right/assert_greater_equal/y:0) = 2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4
Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertNrandom_flip/stateless_random_flip_left_right/assert_greater_equal/All:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0:random_flip/stateless_random_flip_left_right/Rank:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0Lrandom_flip/stateless_random_flip_left_right/assert_greater_equal/y:output:0W^random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2Q
Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert©
?random_flip/stateless_random_flip_left_right/control_dependencyIdentityrandom_flip_inputP^random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertW^random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*$
_class
loc:@random_flip_input*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?random_flip/stateless_random_flip_left_right/control_dependencyÝ
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2M
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0P^random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2d
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterÐ
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2]
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0arandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2`
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2æ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subö
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulÛ
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2G
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniform¯
3random_flip/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?25
3random_flip/stateless_random_flip_left_right/Less/y
1random_flip/stateless_random_flip_left_right/LessLessIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0<random_flip/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: 23
1random_flip/stateless_random_flip_left_right/LessÏ
,random_flip/stateless_random_flip_left_rightStatelessIf5random_flip/stateless_random_flip_left_right/Less:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *L
else_branch=R;
9random_flip_stateless_random_flip_left_right_false_163586*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*K
then_branch<R:
8random_flip_stateless_random_flip_left_right_true_1635852.
,random_flip/stateless_random_flip_left_rightè
5random_flip/stateless_random_flip_left_right/IdentityIdentity5random_flip/stateless_random_flip_left_right:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/stateless_random_flip_left_right/Identity¨
-random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_flip/stateful_uniform_full_int_1/shape¨
-random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_flip/stateful_uniform_full_int_1/Constõ
,random_flip/stateful_uniform_full_int_1/ProdProd6random_flip/stateful_uniform_full_int_1/shape:output:06random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int_1/Prod¢
.random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip/stateful_uniform_full_int_1/Cast/xÏ
.random_flip/stateful_uniform_full_int_1/Cast_1Cast5random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.random_flip/stateful_uniform_full_int_1/Cast_1
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource7random_flip/stateful_uniform_full_int_1/Cast/x:output:02random_flip/stateful_uniform_full_int_1/Cast_1:y:05^random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:28
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipÄ
;random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;random_flip/stateful_uniform_full_int_1/strided_slice/stackÈ
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1È
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Ø
5random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask27
5random_flip/stateful_uniform_full_int_1/strided_sliceÞ
/random_flip/stateful_uniform_full_int_1/BitcastBitcast>random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int_1/BitcastÈ
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackÌ
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Ì
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Ð
7random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Frandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:29
7random_flip/stateful_uniform_full_int_1/strided_slice_1ä
1random_flip/stateful_uniform_full_int_1/Bitcast_1Bitcast@random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type023
1random_flip/stateful_uniform_full_int_1/Bitcast_1
+random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_flip/stateful_uniform_full_int_1/alg
'random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV26random_flip/stateful_uniform_full_int_1/shape:output:0:random_flip/stateful_uniform_full_int_1/Bitcast_1:output:08random_flip/stateful_uniform_full_int_1/Bitcast:output:04random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	2)
'random_flip/stateful_uniform_full_int_1~
random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like_1¹
random_flip/stack_1Pack0random_flip/stateful_uniform_full_int_1:output:0!random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack_1
!random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_flip/strided_slice_1/stack
#random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_flip/strided_slice_1/stack_1
#random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_flip/strided_slice_1/stack_2Ü
random_flip/strided_slice_1StridedSlicerandom_flip/stack_1:output:0*random_flip/strided_slice_1/stack:output:0,random_flip/strided_slice_1/stack_1:output:0,random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_slice_1Ð
/random_flip/stateless_random_flip_up_down/ShapeShape>random_flip/stateless_random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:21
/random_flip/stateless_random_flip_up_down/ShapeÑ
=random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2?
=random_flip/stateless_random_flip_up_down/strided_slice/stackÌ
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Ì
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Ú
7random_flip/stateless_random_flip_up_down/strided_sliceStridedSlice8random_flip/stateless_random_flip_up_down/Shape:output:0Frandom_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask29
7random_flip/stateless_random_flip_up_down/strided_sliceÄ
?random_flip/stateless_random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2A
?random_flip/stateless_random_flip_up_down/assert_positive/ConstÑ
Jrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/LessLessHrandom_flip/stateless_random_flip_up_down/assert_positive/Const:output:0@random_flip/stateless_random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:2L
Jrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Lessä
Krandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Krandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/ConstÛ
Irandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/AllAllNrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Less:z:0Trandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/All
Rrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2T
Rrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Const£
Zrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2\
Zrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0í
Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertRrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/All:output:0crandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg*

T
2*
_output_shapes
 2U
Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert¢
.random_flip/stateless_random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip/stateless_random_flip_up_down/RankÆ
@random_flip/stateless_random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@random_flip/stateless_random_flip_up_down/assert_greater_equal/yÏ
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual7random_flip/stateless_random_flip_up_down/Rank:output:0Irandom_flip/stateless_random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2M
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/GreaterEqualÌ
Crandom_flip/stateless_random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2E
Crandom_flip/stateless_random_flip_up_down/assert_greater_equal/RankÚ
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/startÚ
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/delta§
Drandom_flip/stateless_random_flip_up_down/assert_greater_equal/rangeRangeSrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/start:output:0Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Rank:output:0Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2F
Drandom_flip/stateless_random_flip_up_down/assert_greater_equal/rangeÇ
Brandom_flip/stateless_random_flip_up_down/assert_greater_equal/AllAllOrandom_flip/stateless_random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: 2D
Brandom_flip/stateless_random_flip_up_down/assert_greater_equal/All
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2M
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2O
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_1
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*H
value?B= B7x (random_flip/stateless_random_flip_up_down/Rank:0) = 2O
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_2¨
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Z
valueQBO BIy (random_flip/stateless_random_flip_up_down/assert_greater_equal/y:0) = 2O
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_3
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1¢
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*H
value?B= B7x (random_flip/stateless_random_flip_up_down/Rank:0) = 2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2´
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Z
valueQBO BIy (random_flip/stateless_random_flip_up_down/assert_greater_equal/y:0) = 2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4ì
Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertAssertKrandom_flip/stateless_random_flip_up_down/assert_greater_equal/All:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:07random_flip/stateless_random_flip_up_down/Rank:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0Irandom_flip/stateless_random_flip_up_down/assert_greater_equal/y:output:0T^random_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2N
Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assertî
<random_flip/stateless_random_flip_up_down/control_dependencyIdentity>random_flip/stateless_random_flip_left_right/Identity:output:0M^random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertT^random_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*H
_class>
<:loc:@random_flip/stateless_random_flip_left_right/Identity*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<random_flip/stateless_random_flip_up_down/control_dependency×
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2J
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxù
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip/strided_slice_1:output:0M^random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert* 
_output_shapes
::2a
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterÇ
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Z
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgó
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Qrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0erandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0irandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2]
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2Ú
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubOrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subê
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMuldrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulÏ
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2D
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniform©
0random_flip/stateless_random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0random_flip/stateless_random_flip_up_down/Less/y
.random_flip/stateless_random_flip_up_down/LessLessFrandom_flip/stateless_random_flip_up_down/stateless_random_uniform:z:09random_flip/stateless_random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: 20
.random_flip/stateless_random_flip_up_down/Less½
)random_flip/stateless_random_flip_up_downStatelessIf2random_flip/stateless_random_flip_up_down/Less:z:0Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *I
else_branch:R8
6random_flip_stateless_random_flip_up_down_false_163662*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
then_branch9R7
5random_flip_stateless_random_flip_up_down_true_1636612+
)random_flip/stateless_random_flip_up_downß
2random_flip/stateless_random_flip_up_down/IdentityIdentity2random_flip/stateless_random_flip_up_down:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_up_down/Identity
random_rotation/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
random_rotation/ExpandDims/dimç
random_rotation/ExpandDims
ExpandDims;random_flip/stateless_random_flip_up_down/Identity:output:0'random_rotation/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_rotation/ExpandDims
random_rotation/ShapeShape#random_rotation/ExpandDims:output:0*
T0*
_output_shapes
:2
random_rotation/Shape
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2Â
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice¡
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_1/stack¥
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_1/stack_1
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2Ì
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast¡
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_2/stack¥
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_2/stack_1
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2Ì
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1®
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2&
$random_rotation/stateful_uniform/max
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation/stateful_uniform/ConstÙ
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2'
%random_rotation/stateful_uniform/Prod
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_rotation/stateful_uniform/Cast/xº
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'random_rotation/stateful_uniform/Cast_1©
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:21
/random_rotation/stateful_uniform/RngReadAndSkip¶
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4random_rotation/stateful_uniform/strided_slice/stackº
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_1º
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_2®
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask20
.random_rotation/stateful_uniform/strided_sliceÉ
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02*
(random_rotation/stateful_uniform/Bitcastº
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice_1/stack¾
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_1¾
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_2¦
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:22
0random_rotation/stateful_uniform/strided_slice_1Ï
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation/stateful_uniform/Bitcast_1À
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=random_rotation/stateful_uniform/StatelessRandomUniformV2/alg
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9random_rotation/stateful_uniform/StatelessRandomUniformV2Ò
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subï
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_rotation/stateful_uniform/mulÔ
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/y¾
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub¥
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yÄ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Ó
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/mul¥
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yÂ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2×
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_1×
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_3×
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yê
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yÂ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yÄ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ù
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yÂ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ù
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_3×
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/add×
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yð
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation/rotation_matrix/truediv_1¢
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape´
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack¸
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1¸
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2¢
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice©
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_2¿
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackÃ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2×
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_2¿
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackÃ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2×
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Neg¿
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackÃ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ù
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_3¿
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackÃ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2×
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_3¿
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackÃ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2×
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5¿
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackÃ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Û
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6¢
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Constõ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisÈ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation/rotation_matrix/concat
random_rotation/transform/ShapeShape#random_rotation/ExpandDims:output:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape¨
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack¬
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1¬
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2ê
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_value½
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3#random_rotation/ExpandDims:output:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3Ö
random_rotation/SqueezeSqueezeIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2
random_rotation/Squeeze
IdentityIdentity random_rotation/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÀ
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip7^random_flip/stateful_uniform_full_int_1/RngReadAndSkipP^random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertW^random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterM^random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertT^random_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertY^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter0^random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip/stateful_uniform_full_int_1/RngReadAndSkip6random_flip/stateful_uniform_full_int_1/RngReadAndSkip2¢
Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertOrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert2°
Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertVrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2º
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2È
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterbrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertLrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert2ª
Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertSrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert2´
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgXrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2Â
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip:_ [
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
ÚL
À
I__inference_sequential_17_layer_call_and_return_conditional_losses_161952

inputs!
sequential_1_161899:	!
sequential_1_161901:	*
conv2d_90_161904: 
conv2d_90_161906: *
conv2d_91_161910: @
conv2d_91_161912:@*
conv2d_92_161916:@@
conv2d_92_161918:@*
conv2d_93_161922:@@
conv2d_93_161924:@*
conv2d_94_161928:@@
conv2d_94_161930:@*
conv2d_95_161934:@@
conv2d_95_161936:@"
dense_30_161941:	@
dense_30_161943:@!
dense_31_161946:@
dense_31_161948:
identity¢!conv2d_90/StatefulPartitionedCall¢!conv2d_91/StatefulPartitionedCall¢!conv2d_92/StatefulPartitionedCall¢!conv2d_93/StatefulPartitionedCall¢!conv2d_94/StatefulPartitionedCall¢!conv2d_95/StatefulPartitionedCall¢ dense_30/StatefulPartitionedCall¢ dense_31/StatefulPartitionedCall¢$sequential_1/StatefulPartitionedCallä
sequential/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1618532
sequential/PartitionedCallË
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall#sequential/PartitionedCall:output:0sequential_1_161899sequential_1_161901*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_sequential_1_layer_call_and_return_conditional_losses_1618292&
$sequential_1/StatefulPartitionedCallÊ
!conv2d_90/StatefulPartitionedCallStatefulPartitionedCall-sequential_1/StatefulPartitionedCall:output:0conv2d_90_161904conv2d_90_161906*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿþþ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_90_layer_call_and_return_conditional_losses_1612762#
!conv2d_90/StatefulPartitionedCall
 max_pooling2d_90/PartitionedCallPartitionedCall*conv2d_90/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_1612862"
 max_pooling2d_90/PartitionedCallÄ
!conv2d_91/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_90/PartitionedCall:output:0conv2d_91_161910conv2d_91_161912*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_91_layer_call_and_return_conditional_losses_1612992#
!conv2d_91/StatefulPartitionedCall
 max_pooling2d_91/PartitionedCallPartitionedCall*conv2d_91/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ>>@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_1613092"
 max_pooling2d_91/PartitionedCallÄ
!conv2d_92/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_91/PartitionedCall:output:0conv2d_92_161916conv2d_92_161918*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ<<@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_92_layer_call_and_return_conditional_losses_1613222#
!conv2d_92/StatefulPartitionedCall
 max_pooling2d_92/PartitionedCallPartitionedCall*conv2d_92/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_1613322"
 max_pooling2d_92/PartitionedCallÄ
!conv2d_93/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_92/PartitionedCall:output:0conv2d_93_161922conv2d_93_161924*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_93_layer_call_and_return_conditional_losses_1613452#
!conv2d_93/StatefulPartitionedCall
 max_pooling2d_93/PartitionedCallPartitionedCall*conv2d_93/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_1613552"
 max_pooling2d_93/PartitionedCallÄ
!conv2d_94/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_93/PartitionedCall:output:0conv2d_94_161928conv2d_94_161930*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_94_layer_call_and_return_conditional_losses_1613682#
!conv2d_94/StatefulPartitionedCall
 max_pooling2d_94/PartitionedCallPartitionedCall*conv2d_94/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_1613782"
 max_pooling2d_94/PartitionedCallÄ
!conv2d_95/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_94/PartitionedCall:output:0conv2d_95_161934conv2d_95_161936*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_conv2d_95_layer_call_and_return_conditional_losses_1613912#
!conv2d_95/StatefulPartitionedCall
 max_pooling2d_95/PartitionedCallPartitionedCall*conv2d_95/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_1614012"
 max_pooling2d_95/PartitionedCallþ
flatten_15/PartitionedCallPartitionedCall)max_pooling2d_95/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_flatten_15_layer_call_and_return_conditional_losses_1614092
flatten_15/PartitionedCall±
 dense_30/StatefulPartitionedCallStatefulPartitionedCall#flatten_15/PartitionedCall:output:0dense_30_161941dense_30_161943*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_30_layer_call_and_return_conditional_losses_1614222"
 dense_30/StatefulPartitionedCall·
 dense_31/StatefulPartitionedCallStatefulPartitionedCall)dense_30/StatefulPartitionedCall:output:0dense_31_161946dense_31_161948*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_31_layer_call_and_return_conditional_losses_1614392"
 dense_31/StatefulPartitionedCall
IdentityIdentity)dense_31/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity
NoOpNoOp"^conv2d_90/StatefulPartitionedCall"^conv2d_91/StatefulPartitionedCall"^conv2d_92/StatefulPartitionedCall"^conv2d_93/StatefulPartitionedCall"^conv2d_94/StatefulPartitionedCall"^conv2d_95/StatefulPartitionedCall!^dense_30/StatefulPartitionedCall!^dense_31/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : : : : : : : : : : 2F
!conv2d_90/StatefulPartitionedCall!conv2d_90/StatefulPartitionedCall2F
!conv2d_91/StatefulPartitionedCall!conv2d_91/StatefulPartitionedCall2F
!conv2d_92/StatefulPartitionedCall!conv2d_92/StatefulPartitionedCall2F
!conv2d_93/StatefulPartitionedCall!conv2d_93/StatefulPartitionedCall2F
!conv2d_94/StatefulPartitionedCall!conv2d_94/StatefulPartitionedCall2F
!conv2d_95/StatefulPartitionedCall!conv2d_95/StatefulPartitionedCall2D
 dense_30/StatefulPartitionedCall dense_30/StatefulPartitionedCall2D
 dense_31/StatefulPartitionedCall dense_31/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
þ
E__inference_conv2d_91_layer_call_and_return_conditional_losses_161299

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2
Reluu
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}}@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 
_user_specified_nameinputs
£ý
ç$
"__inference__traced_restore_165048
file_prefix;
!assignvariableop_conv2d_90_kernel: /
!assignvariableop_1_conv2d_90_bias: =
#assignvariableop_2_conv2d_91_kernel: @/
!assignvariableop_3_conv2d_91_bias:@=
#assignvariableop_4_conv2d_92_kernel:@@/
!assignvariableop_5_conv2d_92_bias:@=
#assignvariableop_6_conv2d_93_kernel:@@/
!assignvariableop_7_conv2d_93_bias:@=
#assignvariableop_8_conv2d_94_kernel:@@/
!assignvariableop_9_conv2d_94_bias:@>
$assignvariableop_10_conv2d_95_kernel:@@0
"assignvariableop_11_conv2d_95_bias:@6
#assignvariableop_12_dense_30_kernel:	@/
!assignvariableop_13_dense_30_bias:@5
#assignvariableop_14_dense_31_kernel:@/
!assignvariableop_15_dense_31_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: *
assignvariableop_21_variable:	,
assignvariableop_22_variable_1:	#
assignvariableop_23_total: #
assignvariableop_24_count: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: E
+assignvariableop_27_adam_conv2d_90_kernel_m: 7
)assignvariableop_28_adam_conv2d_90_bias_m: E
+assignvariableop_29_adam_conv2d_91_kernel_m: @7
)assignvariableop_30_adam_conv2d_91_bias_m:@E
+assignvariableop_31_adam_conv2d_92_kernel_m:@@7
)assignvariableop_32_adam_conv2d_92_bias_m:@E
+assignvariableop_33_adam_conv2d_93_kernel_m:@@7
)assignvariableop_34_adam_conv2d_93_bias_m:@E
+assignvariableop_35_adam_conv2d_94_kernel_m:@@7
)assignvariableop_36_adam_conv2d_94_bias_m:@E
+assignvariableop_37_adam_conv2d_95_kernel_m:@@7
)assignvariableop_38_adam_conv2d_95_bias_m:@=
*assignvariableop_39_adam_dense_30_kernel_m:	@6
(assignvariableop_40_adam_dense_30_bias_m:@<
*assignvariableop_41_adam_dense_31_kernel_m:@6
(assignvariableop_42_adam_dense_31_bias_m:E
+assignvariableop_43_adam_conv2d_90_kernel_v: 7
)assignvariableop_44_adam_conv2d_90_bias_v: E
+assignvariableop_45_adam_conv2d_91_kernel_v: @7
)assignvariableop_46_adam_conv2d_91_bias_v:@E
+assignvariableop_47_adam_conv2d_92_kernel_v:@@7
)assignvariableop_48_adam_conv2d_92_bias_v:@E
+assignvariableop_49_adam_conv2d_93_kernel_v:@@7
)assignvariableop_50_adam_conv2d_93_bias_v:@E
+assignvariableop_51_adam_conv2d_94_kernel_v:@@7
)assignvariableop_52_adam_conv2d_94_bias_v:@E
+assignvariableop_53_adam_conv2d_95_kernel_v:@@7
)assignvariableop_54_adam_conv2d_95_bias_v:@=
*assignvariableop_55_adam_dense_30_kernel_v:	@6
(assignvariableop_56_adam_dense_30_bias_v:@<
*assignvariableop_57_adam_dense_31_kernel_v:@6
(assignvariableop_58_adam_dense_31_bias_v:
identity_60¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9¬!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*¸ 
value® B« <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*
valueB<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesÚ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesó
ð::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_90_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¦
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_90_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¨
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_91_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¦
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_91_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¨
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_92_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¦
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_92_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¨
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_93_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¦
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_93_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¨
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_94_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¦
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_94_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10¬
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_95_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ª
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_95_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_dense_30_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_dense_30_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14«
AssignVariableOp_14AssignVariableOp#assignvariableop_14_dense_31_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15©
AssignVariableOp_15AssignVariableOp!assignvariableop_15_dense_31_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16¥
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17§
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18§
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¦
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20®
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21¤
AssignVariableOp_21AssignVariableOpassignvariableop_21_variableIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¦
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¡
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¡
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27³
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_90_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_90_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29³
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_91_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_91_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31³
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_92_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_92_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33³
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_93_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_93_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35³
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_94_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_94_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37³
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_95_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_95_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39²
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_dense_30_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40°
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_dense_30_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41²
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_dense_31_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42°
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_dense_31_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43³
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_90_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_90_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45³
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_91_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_91_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47³
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_92_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_92_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49³
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_93_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_93_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51³
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_94_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_94_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53³
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_95_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_95_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55²
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_dense_30_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56°
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_dense_30_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57²
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_dense_31_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58°
AssignVariableOp_58AssignVariableOp(assignvariableop_58_adam_dense_31_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpð

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59f
Identity_60IdentityIdentity_59:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_60Ø

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_60Identity_60:output:0*
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù
M
1__inference_max_pooling2d_95_layer_call_fn_164245

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_1612292
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
î
H__inference_sequential_1_layer_call_and_return_conditional_losses_163510

inputsK
=random_flip_stateful_uniform_full_int_rngreadandskip_resource:	F
8random_rotation_stateful_uniform_rngreadandskip_resource:	
identity¢4random_flip/stateful_uniform_full_int/RngReadAndSkip¢6random_flip/stateful_uniform_full_int_1/RngReadAndSkip¢Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert¢Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¢[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg¢brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter¢Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert¢Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert¢Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg¢_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter¢/random_rotation/stateful_uniform/RngReadAndSkip¤
+random_flip/stateful_uniform_full_int/shapeConst*
_output_shapes
:*
dtype0*
valueB:2-
+random_flip/stateful_uniform_full_int/shape¤
+random_flip/stateful_uniform_full_int/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+random_flip/stateful_uniform_full_int/Constí
*random_flip/stateful_uniform_full_int/ProdProd4random_flip/stateful_uniform_full_int/shape:output:04random_flip/stateful_uniform_full_int/Const:output:0*
T0*
_output_shapes
: 2,
*random_flip/stateful_uniform_full_int/Prod
,random_flip/stateful_uniform_full_int/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2.
,random_flip/stateful_uniform_full_int/Cast/xÉ
,random_flip/stateful_uniform_full_int/Cast_1Cast3random_flip/stateful_uniform_full_int/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int/Cast_1Â
4random_flip/stateful_uniform_full_int/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource5random_flip/stateful_uniform_full_int/Cast/x:output:00random_flip/stateful_uniform_full_int/Cast_1:y:0*
_output_shapes
:26
4random_flip/stateful_uniform_full_int/RngReadAndSkipÀ
9random_flip/stateful_uniform_full_int/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9random_flip/stateful_uniform_full_int/strided_slice/stackÄ
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_1Ä
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice/stack_2Ì
3random_flip/stateful_uniform_full_int/strided_sliceStridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Brandom_flip/stateful_uniform_full_int/strided_slice/stack:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_1:output:0Drandom_flip/stateful_uniform_full_int/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask25
3random_flip/stateful_uniform_full_int/strided_sliceØ
-random_flip/stateful_uniform_full_int/BitcastBitcast<random_flip/stateful_uniform_full_int/strided_slice:output:0*
T0	*
_output_shapes
:*

type02/
-random_flip/stateful_uniform_full_int/BitcastÄ
;random_flip/stateful_uniform_full_int/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2=
;random_flip/stateful_uniform_full_int/strided_slice_1/stackÈ
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_1È
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int/strided_slice_1/stack_2Ä
5random_flip/stateful_uniform_full_int/strided_slice_1StridedSlice<random_flip/stateful_uniform_full_int/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int/strided_slice_1/stack:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_1:output:0Frandom_flip/stateful_uniform_full_int/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:27
5random_flip/stateful_uniform_full_int/strided_slice_1Þ
/random_flip/stateful_uniform_full_int/Bitcast_1Bitcast>random_flip/stateful_uniform_full_int/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int/Bitcast_1
)random_flip/stateful_uniform_full_int/algConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_flip/stateful_uniform_full_int/algö
%random_flip/stateful_uniform_full_intStatelessRandomUniformFullIntV24random_flip/stateful_uniform_full_int/shape:output:08random_flip/stateful_uniform_full_int/Bitcast_1:output:06random_flip/stateful_uniform_full_int/Bitcast:output:02random_flip/stateful_uniform_full_int/alg:output:0*
_output_shapes
:*
dtype0	2'
%random_flip/stateful_uniform_full_intz
random_flip/zeros_likeConst*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like±
random_flip/stackPack.random_flip/stateful_uniform_full_int:output:0random_flip/zeros_like:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack
random_flip/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2!
random_flip/strided_slice/stack
!random_flip/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!random_flip/strided_slice/stack_1
!random_flip/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!random_flip/strided_slice/stack_2Ð
random_flip/strided_sliceStridedSlicerandom_flip/stack:output:0(random_flip/strided_slice/stack:output:0*random_flip/strided_slice/stack_1:output:0*random_flip/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_slice
2random_flip/stateless_random_flip_left_right/ShapeShapeinputs*
T0*
_output_shapes
:24
2random_flip/stateless_random_flip_left_right/Shape×
@random_flip/stateless_random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2B
@random_flip/stateless_random_flip_left_right/strided_slice/stackÒ
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_1Ò
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2D
Brandom_flip/stateless_random_flip_left_right/strided_slice/stack_2ì
:random_flip/stateless_random_flip_left_right/strided_sliceStridedSlice;random_flip/stateless_random_flip_left_right/Shape:output:0Irandom_flip/stateless_random_flip_left_right/strided_slice/stack:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_1:output:0Krandom_flip/stateless_random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2<
:random_flip/stateless_random_flip_left_right/strided_sliceÊ
Brandom_flip/stateless_random_flip_left_right/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2D
Brandom_flip/stateless_random_flip_left_right/assert_positive/ConstÝ
Mrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/LessLessKrandom_flip/stateless_random_flip_left_right/assert_positive/Const:output:0Crandom_flip/stateless_random_flip_left_right/strided_slice:output:0*
T0*
_output_shapes
:2O
Mrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Lessê
Nrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2P
Nrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Constç
Lrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/AllAllQrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Less:z:0Wrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2N
Lrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/All
Urandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2W
Urandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Const©
]random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2_
]random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0
Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertAssertUrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/All:output:0frandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert/data_0:output:0*

T
2*
_output_shapes
 2X
Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert¨
1random_flip/stateless_random_flip_left_right/RankConst*
_output_shapes
: *
dtype0*
value	B :23
1random_flip/stateless_random_flip_left_right/RankÌ
Crandom_flip/stateless_random_flip_left_right/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2E
Crandom_flip/stateless_random_flip_left_right/assert_greater_equal/yÛ
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/GreaterEqualGreaterEqual:random_flip/stateless_random_flip_left_right/Rank:output:0Lrandom_flip/stateless_random_flip_left_right/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2P
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/GreaterEqualÒ
Frandom_flip/stateless_random_flip_left_right/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2H
Frandom_flip/stateless_random_flip_left_right/assert_greater_equal/Rankà
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2O
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/startà
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2O
Mrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/delta¶
Grandom_flip/stateless_random_flip_left_right/assert_greater_equal/rangeRangeVrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/start:output:0Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Rank:output:0Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2I
Grandom_flip/stateless_random_flip_left_right/assert_greater_equal/rangeÓ
Erandom_flip/stateless_random_flip_left_right/assert_greater_equal/AllAllRrandom_flip/stateless_random_flip_left_right/assert_greater_equal/GreaterEqual:z:0Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/range:output:0*
_output_shapes
: 2G
Erandom_flip/stateless_random_flip_left_right/assert_greater_equal/All
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2P
Nrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2R
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_1
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (random_flip/stateless_random_flip_left_right/Rank:0) = 2R
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_2±
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*]
valueTBR BLy (random_flip/stateless_random_flip_left_right/assert_greater_equal/y:0) = 2R
Prandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Const_3
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1«
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*K
valueBB@ B:x (random_flip/stateless_random_flip_left_right/Rank:0) = 2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2½
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*]
valueTBR BLy (random_flip/stateless_random_flip_left_right/assert_greater_equal/y:0) = 2X
Vrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4
Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertAssertNrandom_flip/stateless_random_flip_left_right/assert_greater_equal/All:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_0:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_1:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_2:output:0:random_flip/stateless_random_flip_left_right/Rank:output:0_random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert/data_4:output:0Lrandom_flip/stateless_random_flip_left_right/assert_greater_equal/y:output:0W^random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2Q
Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert
?random_flip/stateless_random_flip_left_right/control_dependencyIdentityinputsP^random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertW^random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert*
T0*
_class
loc:@inputs*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2A
?random_flip/stateless_random_flip_left_right/control_dependencyÝ
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2M
Krandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shapeÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/minÛ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter"random_flip/strided_slice:output:0P^random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert* 
_output_shapes
::2d
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterÐ
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2]
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Trandom_flip/stateless_random_flip_left_right/stateless_random_uniform/shape:output:0hrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0arandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2`
^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2æ
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subSubRrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/max:output:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/subö
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulMulgrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomUniformV2:output:0Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mulÛ
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniformAddV2Mrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/mul:z:0Rrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2G
Erandom_flip/stateless_random_flip_left_right/stateless_random_uniform¯
3random_flip/stateless_random_flip_left_right/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?25
3random_flip/stateless_random_flip_left_right/Less/y
1random_flip/stateless_random_flip_left_right/LessLessIrandom_flip/stateless_random_flip_left_right/stateless_random_uniform:z:0<random_flip/stateless_random_flip_left_right/Less/y:output:0*
T0*
_output_shapes
: 23
1random_flip/stateless_random_flip_left_right/LessÏ
,random_flip/stateless_random_flip_left_rightStatelessIf5random_flip/stateless_random_flip_left_right/Less:z:0Hrandom_flip/stateless_random_flip_left_right/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *L
else_branch=R;
9random_flip_stateless_random_flip_left_right_false_163305*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*K
then_branch<R:
8random_flip_stateless_random_flip_left_right_true_1633042.
,random_flip/stateless_random_flip_left_rightè
5random_flip/stateless_random_flip_left_right/IdentityIdentity5random_flip/stateless_random_flip_left_right:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5random_flip/stateless_random_flip_left_right/Identity¨
-random_flip/stateful_uniform_full_int_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_flip/stateful_uniform_full_int_1/shape¨
-random_flip/stateful_uniform_full_int_1/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-random_flip/stateful_uniform_full_int_1/Constõ
,random_flip/stateful_uniform_full_int_1/ProdProd6random_flip/stateful_uniform_full_int_1/shape:output:06random_flip/stateful_uniform_full_int_1/Const:output:0*
T0*
_output_shapes
: 2.
,random_flip/stateful_uniform_full_int_1/Prod¢
.random_flip/stateful_uniform_full_int_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip/stateful_uniform_full_int_1/Cast/xÏ
.random_flip/stateful_uniform_full_int_1/Cast_1Cast5random_flip/stateful_uniform_full_int_1/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 20
.random_flip/stateful_uniform_full_int_1/Cast_1
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipRngReadAndSkip=random_flip_stateful_uniform_full_int_rngreadandskip_resource7random_flip/stateful_uniform_full_int_1/Cast/x:output:02random_flip/stateful_uniform_full_int_1/Cast_1:y:05^random_flip/stateful_uniform_full_int/RngReadAndSkip*
_output_shapes
:28
6random_flip/stateful_uniform_full_int_1/RngReadAndSkipÄ
;random_flip/stateful_uniform_full_int_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;random_flip/stateful_uniform_full_int_1/strided_slice/stackÈ
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_1È
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice/stack_2Ø
5random_flip/stateful_uniform_full_int_1/strided_sliceStridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Drandom_flip/stateful_uniform_full_int_1/strided_slice/stack:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_1:output:0Frandom_flip/stateful_uniform_full_int_1/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask27
5random_flip/stateful_uniform_full_int_1/strided_sliceÞ
/random_flip/stateful_uniform_full_int_1/BitcastBitcast>random_flip/stateful_uniform_full_int_1/strided_slice:output:0*
T0	*
_output_shapes
:*

type021
/random_flip/stateful_uniform_full_int_1/BitcastÈ
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2?
=random_flip/stateful_uniform_full_int_1/strided_slice_1/stackÌ
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1Ì
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2Ð
7random_flip/stateful_uniform_full_int_1/strided_slice_1StridedSlice>random_flip/stateful_uniform_full_int_1/RngReadAndSkip:value:0Frandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_1:output:0Hrandom_flip/stateful_uniform_full_int_1/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:29
7random_flip/stateful_uniform_full_int_1/strided_slice_1ä
1random_flip/stateful_uniform_full_int_1/Bitcast_1Bitcast@random_flip/stateful_uniform_full_int_1/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type023
1random_flip/stateful_uniform_full_int_1/Bitcast_1
+random_flip/stateful_uniform_full_int_1/algConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_flip/stateful_uniform_full_int_1/alg
'random_flip/stateful_uniform_full_int_1StatelessRandomUniformFullIntV26random_flip/stateful_uniform_full_int_1/shape:output:0:random_flip/stateful_uniform_full_int_1/Bitcast_1:output:08random_flip/stateful_uniform_full_int_1/Bitcast:output:04random_flip/stateful_uniform_full_int_1/alg:output:0*
_output_shapes
:*
dtype0	2)
'random_flip/stateful_uniform_full_int_1~
random_flip/zeros_like_1Const*
_output_shapes
:*
dtype0	*
valueB	R 2
random_flip/zeros_like_1¹
random_flip/stack_1Pack0random_flip/stateful_uniform_full_int_1:output:0!random_flip/zeros_like_1:output:0*
N*
T0	*
_output_shapes

:2
random_flip/stack_1
!random_flip/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!random_flip/strided_slice_1/stack
#random_flip/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#random_flip/strided_slice_1/stack_1
#random_flip/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#random_flip/strided_slice_1/stack_2Ü
random_flip/strided_slice_1StridedSlicerandom_flip/stack_1:output:0*random_flip/strided_slice_1/stack:output:0,random_flip/strided_slice_1/stack_1:output:0,random_flip/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask*
end_mask*
shrink_axis_mask2
random_flip/strided_slice_1Ð
/random_flip/stateless_random_flip_up_down/ShapeShape>random_flip/stateless_random_flip_left_right/Identity:output:0*
T0*
_output_shapes
:21
/random_flip/stateless_random_flip_up_down/ShapeÑ
=random_flip/stateless_random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2?
=random_flip/stateless_random_flip_up_down/strided_slice/stackÌ
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_1Ì
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?random_flip/stateless_random_flip_up_down/strided_slice/stack_2Ú
7random_flip/stateless_random_flip_up_down/strided_sliceStridedSlice8random_flip/stateless_random_flip_up_down/Shape:output:0Frandom_flip/stateless_random_flip_up_down/strided_slice/stack:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_1:output:0Hrandom_flip/stateless_random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask29
7random_flip/stateless_random_flip_up_down/strided_sliceÄ
?random_flip/stateless_random_flip_up_down/assert_positive/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2A
?random_flip/stateless_random_flip_up_down/assert_positive/ConstÑ
Jrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/LessLessHrandom_flip/stateless_random_flip_up_down/assert_positive/Const:output:0@random_flip/stateless_random_flip_up_down/strided_slice:output:0*
T0*
_output_shapes
:2L
Jrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Lessä
Krandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2M
Krandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/ConstÛ
Irandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/AllAllNrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Less:z:0Trandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Const:output:0*
_output_shapes
: 2K
Irandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/All
Rrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/ConstConst*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2T
Rrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Const£
Zrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*;
value2B0 B*inner 3 dims of 'image.shape' must be > 0.2\
Zrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0í
Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertAssertRrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/All:output:0crandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert/data_0:output:0\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg*

T
2*
_output_shapes
 2U
Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert¢
.random_flip/stateless_random_flip_up_down/RankConst*
_output_shapes
: *
dtype0*
value	B :20
.random_flip/stateless_random_flip_up_down/RankÆ
@random_flip/stateless_random_flip_up_down/assert_greater_equal/yConst*
_output_shapes
: *
dtype0*
value	B :2B
@random_flip/stateless_random_flip_up_down/assert_greater_equal/yÏ
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/GreaterEqualGreaterEqual7random_flip/stateless_random_flip_up_down/Rank:output:0Irandom_flip/stateless_random_flip_up_down/assert_greater_equal/y:output:0*
T0*
_output_shapes
: 2M
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/GreaterEqualÌ
Crandom_flip/stateless_random_flip_up_down/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2E
Crandom_flip/stateless_random_flip_up_down/assert_greater_equal/RankÚ
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2L
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/startÚ
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2L
Jrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/delta§
Drandom_flip/stateless_random_flip_up_down/assert_greater_equal/rangeRangeSrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/start:output:0Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Rank:output:0Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2F
Drandom_flip/stateless_random_flip_up_down/assert_greater_equal/rangeÇ
Brandom_flip/stateless_random_flip_up_down/assert_greater_equal/AllAllOrandom_flip/stateless_random_flip_up_down/assert_greater_equal/GreaterEqual:z:0Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/range:output:0*
_output_shapes
: 2D
Brandom_flip/stateless_random_flip_up_down/assert_greater_equal/All
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2M
Krandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2O
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_1
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*H
value?B= B7x (random_flip/stateless_random_flip_up_down/Rank:0) = 2O
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_2¨
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*Z
valueQBO BIy (random_flip/stateless_random_flip_up_down/assert_greater_equal/y:0) = 2O
Mrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Const_3
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0Const*
_output_shapes
: *
dtype0*<
value3B1 B+'image' must be at least three-dimensional.2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1¢
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2Const*
_output_shapes
: *
dtype0*H
value?B= B7x (random_flip/stateless_random_flip_up_down/Rank:0) = 2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2´
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4Const*
_output_shapes
: *
dtype0*Z
valueQBO BIy (random_flip/stateless_random_flip_up_down/assert_greater_equal/y:0) = 2U
Srandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4ì
Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertAssertKrandom_flip/stateless_random_flip_up_down/assert_greater_equal/All:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_0:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_1:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_2:output:07random_flip/stateless_random_flip_up_down/Rank:output:0\random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert/data_4:output:0Irandom_flip/stateless_random_flip_up_down/assert_greater_equal/y:output:0T^random_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T

2*
_output_shapes
 2N
Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assertî
<random_flip/stateless_random_flip_up_down/control_dependencyIdentity>random_flip/stateless_random_flip_left_right/Identity:output:0M^random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertT^random_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert*
T0*H
_class>
<:loc:@random_flip/stateless_random_flip_left_right/Identity*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<random_flip/stateless_random_flip_up_down/control_dependency×
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeConst*
_output_shapes
: *
dtype0*
valueB 2J
Hrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shapeÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/minÕ
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/maxù
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter$random_flip/strided_slice_1:output:0M^random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert* 
_output_shapes
::2a
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounterÇ
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgStatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter*
_output_shapes
: 2Z
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgó
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2StatelessRandomUniformV2Qrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/shape:output:0erandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0irandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg:alg:0*
_output_shapes
: 2]
[random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2Ú
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subSubOrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/max:output:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/subê
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulMuldrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomUniformV2:output:0Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/sub:z:0*
T0*
_output_shapes
: 2H
Frandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mulÏ
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniformAddV2Jrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/mul:z:0Orandom_flip/stateless_random_flip_up_down/stateless_random_uniform/min:output:0*
T0*
_output_shapes
: 2D
Brandom_flip/stateless_random_flip_up_down/stateless_random_uniform©
0random_flip/stateless_random_flip_up_down/Less/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?22
0random_flip/stateless_random_flip_up_down/Less/y
.random_flip/stateless_random_flip_up_down/LessLessFrandom_flip/stateless_random_flip_up_down/stateless_random_uniform:z:09random_flip/stateless_random_flip_up_down/Less/y:output:0*
T0*
_output_shapes
: 20
.random_flip/stateless_random_flip_up_down/Less½
)random_flip/stateless_random_flip_up_downStatelessIf2random_flip/stateless_random_flip_up_down/Less:z:0Erandom_flip/stateless_random_flip_up_down/control_dependency:output:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *I
else_branch:R8
6random_flip_stateless_random_flip_up_down_false_163381*+
output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
then_branch9R7
5random_flip_stateless_random_flip_up_down_true_1633802+
)random_flip/stateless_random_flip_up_downß
2random_flip/stateless_random_flip_up_down/IdentityIdentity2random_flip/stateless_random_flip_up_down:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2random_flip/stateless_random_flip_up_down/Identity
random_rotation/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
random_rotation/ExpandDims/dimç
random_rotation/ExpandDims
ExpandDims;random_flip/stateless_random_flip_up_down/Identity:output:0'random_rotation/ExpandDims/dim:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
random_rotation/ExpandDims
random_rotation/ShapeShape#random_rotation/ExpandDims:output:0*
T0*
_output_shapes
:2
random_rotation/Shape
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#random_rotation/strided_slice/stack
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_1
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%random_rotation/strided_slice/stack_2Â
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
random_rotation/strided_slice¡
%random_rotation/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_1/stack¥
'random_rotation/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_1/stack_1
'random_rotation/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_1/stack_2Ì
random_rotation/strided_slice_1StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_1/stack:output:00random_rotation/strided_slice_1/stack_1:output:00random_rotation/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_1
random_rotation/CastCast(random_rotation/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast¡
%random_rotation/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ2'
%random_rotation/strided_slice_2/stack¥
'random_rotation/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/strided_slice_2/stack_1
'random_rotation/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation/strided_slice_2/stack_2Ì
random_rotation/strided_slice_2StridedSlicerandom_rotation/Shape:output:0.random_rotation/strided_slice_2/stack:output:00random_rotation/strided_slice_2/stack_1:output:00random_rotation/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation/strided_slice_2
random_rotation/Cast_1Cast(random_rotation/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation/Cast_1®
&random_rotation/stateful_uniform/shapePack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:2(
&random_rotation/stateful_uniform/shape
$random_rotation/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿2&
$random_rotation/stateful_uniform/min
$random_rotation/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?2&
$random_rotation/stateful_uniform/max
&random_rotation/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2(
&random_rotation/stateful_uniform/ConstÙ
%random_rotation/stateful_uniform/ProdProd/random_rotation/stateful_uniform/shape:output:0/random_rotation/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2'
%random_rotation/stateful_uniform/Prod
'random_rotation/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2)
'random_rotation/stateful_uniform/Cast/xº
'random_rotation/stateful_uniform/Cast_1Cast.random_rotation/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2)
'random_rotation/stateful_uniform/Cast_1©
/random_rotation/stateful_uniform/RngReadAndSkipRngReadAndSkip8random_rotation_stateful_uniform_rngreadandskip_resource0random_rotation/stateful_uniform/Cast/x:output:0+random_rotation/stateful_uniform/Cast_1:y:0*
_output_shapes
:21
/random_rotation/stateful_uniform/RngReadAndSkip¶
4random_rotation/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4random_rotation/stateful_uniform/strided_slice/stackº
6random_rotation/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_1º
6random_rotation/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice/stack_2®
.random_rotation/stateful_uniform/strided_sliceStridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0=random_rotation/stateful_uniform/strided_slice/stack:output:0?random_rotation/stateful_uniform/strided_slice/stack_1:output:0?random_rotation/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask20
.random_rotation/stateful_uniform/strided_sliceÉ
(random_rotation/stateful_uniform/BitcastBitcast7random_rotation/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02*
(random_rotation/stateful_uniform/Bitcastº
6random_rotation/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:28
6random_rotation/stateful_uniform/strided_slice_1/stack¾
8random_rotation/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_1¾
8random_rotation/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation/stateful_uniform/strided_slice_1/stack_2¦
0random_rotation/stateful_uniform/strided_slice_1StridedSlice7random_rotation/stateful_uniform/RngReadAndSkip:value:0?random_rotation/stateful_uniform/strided_slice_1/stack:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_1:output:0Arandom_rotation/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:22
0random_rotation/stateful_uniform/strided_slice_1Ï
*random_rotation/stateful_uniform/Bitcast_1Bitcast9random_rotation/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation/stateful_uniform/Bitcast_1À
=random_rotation/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2?
=random_rotation/stateful_uniform/StatelessRandomUniformV2/alg
9random_rotation/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2/random_rotation/stateful_uniform/shape:output:03random_rotation/stateful_uniform/Bitcast_1:output:01random_rotation/stateful_uniform/Bitcast:output:0Frandom_rotation/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9random_rotation/stateful_uniform/StatelessRandomUniformV2Ò
$random_rotation/stateful_uniform/subSub-random_rotation/stateful_uniform/max:output:0-random_rotation/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2&
$random_rotation/stateful_uniform/subï
$random_rotation/stateful_uniform/mulMulBrandom_rotation/stateful_uniform/StatelessRandomUniformV2:output:0(random_rotation/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2&
$random_rotation/stateful_uniform/mulÔ
 random_rotation/stateful_uniformAddV2(random_rotation/stateful_uniform/mul:z:0-random_rotation/stateful_uniform/min:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2"
 random_rotation/stateful_uniform
%random_rotation/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2'
%random_rotation/rotation_matrix/sub/y¾
#random_rotation/rotation_matrix/subSubrandom_rotation/Cast_1:y:0.random_rotation/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2%
#random_rotation/rotation_matrix/sub¥
#random_rotation/rotation_matrix/CosCos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Cos
'random_rotation/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_1/yÄ
%random_rotation/rotation_matrix/sub_1Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_1Ó
#random_rotation/rotation_matrix/mulMul'random_rotation/rotation_matrix/Cos:y:0)random_rotation/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/mul¥
#random_rotation/rotation_matrix/SinSin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Sin
'random_rotation/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_2/yÂ
%random_rotation/rotation_matrix/sub_2Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_2×
%random_rotation/rotation_matrix/mul_1Mul'random_rotation/rotation_matrix/Sin:y:0)random_rotation/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_1×
%random_rotation/rotation_matrix/sub_3Sub'random_rotation/rotation_matrix/mul:z:0)random_rotation/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_3×
%random_rotation/rotation_matrix/sub_4Sub'random_rotation/rotation_matrix/sub:z:0)random_rotation/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_4
)random_rotation/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2+
)random_rotation/rotation_matrix/truediv/yê
'random_rotation/rotation_matrix/truedivRealDiv)random_rotation/rotation_matrix/sub_4:z:02random_rotation/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2)
'random_rotation/rotation_matrix/truediv
'random_rotation/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_5/yÂ
%random_rotation/rotation_matrix/sub_5Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_5©
%random_rotation/rotation_matrix/Sin_1Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_1
'random_rotation/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_6/yÄ
%random_rotation/rotation_matrix/sub_6Subrandom_rotation/Cast_1:y:00random_rotation/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_6Ù
%random_rotation/rotation_matrix/mul_2Mul)random_rotation/rotation_matrix/Sin_1:y:0)random_rotation/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_2©
%random_rotation/rotation_matrix/Cos_1Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_1
'random_rotation/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'random_rotation/rotation_matrix/sub_7/yÂ
%random_rotation/rotation_matrix/sub_7Subrandom_rotation/Cast:y:00random_rotation/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation/rotation_matrix/sub_7Ù
%random_rotation/rotation_matrix/mul_3Mul)random_rotation/rotation_matrix/Cos_1:y:0)random_rotation/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/mul_3×
#random_rotation/rotation_matrix/addAddV2)random_rotation/rotation_matrix/mul_2:z:0)random_rotation/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/add×
%random_rotation/rotation_matrix/sub_8Sub)random_rotation/rotation_matrix/sub_5:z:0'random_rotation/rotation_matrix/add:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/sub_8
+random_rotation/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation/rotation_matrix/truediv_1/yð
)random_rotation/rotation_matrix/truediv_1RealDiv)random_rotation/rotation_matrix/sub_8:z:04random_rotation/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)random_rotation/rotation_matrix/truediv_1¢
%random_rotation/rotation_matrix/ShapeShape$random_rotation/stateful_uniform:z:0*
T0*
_output_shapes
:2'
%random_rotation/rotation_matrix/Shape´
3random_rotation/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3random_rotation/rotation_matrix/strided_slice/stack¸
5random_rotation/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_1¸
5random_rotation/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5random_rotation/rotation_matrix/strided_slice/stack_2¢
-random_rotation/rotation_matrix/strided_sliceStridedSlice.random_rotation/rotation_matrix/Shape:output:0<random_rotation/rotation_matrix/strided_slice/stack:output:0>random_rotation/rotation_matrix/strided_slice/stack_1:output:0>random_rotation/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2/
-random_rotation/rotation_matrix/strided_slice©
%random_rotation/rotation_matrix/Cos_2Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_2¿
5random_rotation/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_1/stackÃ
7random_rotation/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_1/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_1/stack_2×
/random_rotation/rotation_matrix/strided_slice_1StridedSlice)random_rotation/rotation_matrix/Cos_2:y:0>random_rotation/rotation_matrix/strided_slice_1/stack:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_1©
%random_rotation/rotation_matrix/Sin_2Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_2¿
5random_rotation/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_2/stackÃ
7random_rotation/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_2/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_2/stack_2×
/random_rotation/rotation_matrix/strided_slice_2StridedSlice)random_rotation/rotation_matrix/Sin_2:y:0>random_rotation/rotation_matrix/strided_slice_2/stack:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_2½
#random_rotation/rotation_matrix/NegNeg8random_rotation/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#random_rotation/rotation_matrix/Neg¿
5random_rotation/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_3/stackÃ
7random_rotation/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_3/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_3/stack_2Ù
/random_rotation/rotation_matrix/strided_slice_3StridedSlice+random_rotation/rotation_matrix/truediv:z:0>random_rotation/rotation_matrix/strided_slice_3/stack:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_3©
%random_rotation/rotation_matrix/Sin_3Sin$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Sin_3¿
5random_rotation/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_4/stackÃ
7random_rotation/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_4/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_4/stack_2×
/random_rotation/rotation_matrix/strided_slice_4StridedSlice)random_rotation/rotation_matrix/Sin_3:y:0>random_rotation/rotation_matrix/strided_slice_4/stack:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_4©
%random_rotation/rotation_matrix/Cos_3Cos$random_rotation/stateful_uniform:z:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/Cos_3¿
5random_rotation/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_5/stackÃ
7random_rotation/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_5/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_5/stack_2×
/random_rotation/rotation_matrix/strided_slice_5StridedSlice)random_rotation/rotation_matrix/Cos_3:y:0>random_rotation/rotation_matrix/strided_slice_5/stack:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_5¿
5random_rotation/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        27
5random_rotation/rotation_matrix/strided_slice_6/stackÃ
7random_rotation/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation/rotation_matrix/strided_slice_6/stack_1Ã
7random_rotation/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7random_rotation/rotation_matrix/strided_slice_6/stack_2Û
/random_rotation/rotation_matrix/strided_slice_6StridedSlice-random_rotation/rotation_matrix/truediv_1:z:0>random_rotation/rotation_matrix/strided_slice_6/stack:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_1:output:0@random_rotation/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask21
/random_rotation/rotation_matrix/strided_slice_6¢
.random_rotation/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :20
.random_rotation/rotation_matrix/zeros/packed/1
,random_rotation/rotation_matrix/zeros/packedPack6random_rotation/rotation_matrix/strided_slice:output:07random_rotation/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2.
,random_rotation/rotation_matrix/zeros/packed
+random_rotation/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+random_rotation/rotation_matrix/zeros/Constõ
%random_rotation/rotation_matrix/zerosFill5random_rotation/rotation_matrix/zeros/packed:output:04random_rotation/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%random_rotation/rotation_matrix/zeros
+random_rotation/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2-
+random_rotation/rotation_matrix/concat/axisÈ
&random_rotation/rotation_matrix/concatConcatV28random_rotation/rotation_matrix/strided_slice_1:output:0'random_rotation/rotation_matrix/Neg:y:08random_rotation/rotation_matrix/strided_slice_3:output:08random_rotation/rotation_matrix/strided_slice_4:output:08random_rotation/rotation_matrix/strided_slice_5:output:08random_rotation/rotation_matrix/strided_slice_6:output:0.random_rotation/rotation_matrix/zeros:output:04random_rotation/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2(
&random_rotation/rotation_matrix/concat
random_rotation/transform/ShapeShape#random_rotation/ExpandDims:output:0*
T0*
_output_shapes
:2!
random_rotation/transform/Shape¨
-random_rotation/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-random_rotation/transform/strided_slice/stack¬
/random_rotation/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_1¬
/random_rotation/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation/transform/strided_slice/stack_2ê
'random_rotation/transform/strided_sliceStridedSlice(random_rotation/transform/Shape:output:06random_rotation/transform/strided_slice/stack:output:08random_rotation/transform/strided_slice/stack_1:output:08random_rotation/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2)
'random_rotation/transform/strided_slice
$random_rotation/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$random_rotation/transform/fill_value½
4random_rotation/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3#random_rotation/ExpandDims:output:0/random_rotation/rotation_matrix/concat:output:00random_rotation/transform/strided_slice:output:0-random_rotation/transform/fill_value:output:0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR26
4random_rotation/transform/ImageProjectiveTransformV3Ö
random_rotation/SqueezeSqueezeIrandom_rotation/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
 2
random_rotation/Squeeze
IdentityIdentity random_rotation/Squeeze:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

IdentityÀ
NoOpNoOp5^random_flip/stateful_uniform_full_int/RngReadAndSkip7^random_flip/stateful_uniform_full_int_1/RngReadAndSkipP^random_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertW^random_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert\^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlgc^random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterM^random_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertT^random_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertY^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg`^random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter0^random_rotation/stateful_uniform/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 2l
4random_flip/stateful_uniform_full_int/RngReadAndSkip4random_flip/stateful_uniform_full_int/RngReadAndSkip2p
6random_flip/stateful_uniform_full_int_1/RngReadAndSkip6random_flip/stateful_uniform_full_int_1/RngReadAndSkip2¢
Orandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/AssertOrandom_flip/stateless_random_flip_left_right/assert_greater_equal/Assert/Assert2°
Vrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/AssertVrandom_flip/stateless_random_flip_left_right/assert_positive/assert_less/Assert/Assert2º
[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg[random_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetAlg2È
brandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounterbrandom_flip/stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter2
Lrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/AssertLrandom_flip/stateless_random_flip_up_down/assert_greater_equal/Assert/Assert2ª
Srandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/AssertSrandom_flip/stateless_random_flip_up_down/assert_positive/assert_less/Assert/Assert2´
Xrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlgXrandom_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetAlg2Â
_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter_random_flip/stateless_random_flip_up_down/stateless_random_uniform/StatelessRandomGetKeyCounter2b
/random_rotation/stateful_uniform/RngReadAndSkip/random_rotation/stateful_uniform/RngReadAndSkip:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¨L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ç
serving_default³
W
sequential_inputC
"serving_default_sequential_input:0ÿÿÿÿÿÿÿÿÿ<
dense_310
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Æ

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
	optimizer
trainable_variables
	variables
regularization_losses
	keras_api

signatures
¢_default_save_signature
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_sequential
Æ
layer-0
layer-1
trainable_variables
	variables
regularization_losses
	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_sequential
Æ
layer-0
layer-1
 trainable_variables
!	variables
"regularization_losses
#	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_sequential
½

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
§
*trainable_variables
+	variables
,regularization_losses
-	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layer
½

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
­__call__
+®&call_and_return_all_conditional_losses"
_tf_keras_layer
§
4trainable_variables
5	variables
6regularization_losses
7	keras_api
¯__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layer
½

8kernel
9bias
:trainable_variables
;	variables
<regularization_losses
=	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layer
§
>trainable_variables
?	variables
@regularization_losses
A	keras_api
³__call__
+´&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Bkernel
Cbias
Dtrainable_variables
E	variables
Fregularization_losses
G	keras_api
µ__call__
+¶&call_and_return_all_conditional_losses"
_tf_keras_layer
§
Htrainable_variables
I	variables
Jregularization_losses
K	keras_api
·__call__
+¸&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Lkernel
Mbias
Ntrainable_variables
O	variables
Pregularization_losses
Q	keras_api
¹__call__
+º&call_and_return_all_conditional_losses"
_tf_keras_layer
§
Rtrainable_variables
S	variables
Tregularization_losses
U	keras_api
»__call__
+¼&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Vkernel
Wbias
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
½__call__
+¾&call_and_return_all_conditional_losses"
_tf_keras_layer
§
\trainable_variables
]	variables
^regularization_losses
_	keras_api
¿__call__
+À&call_and_return_all_conditional_losses"
_tf_keras_layer
§
`trainable_variables
a	variables
bregularization_losses
c	keras_api
Á__call__
+Â&call_and_return_all_conditional_losses"
_tf_keras_layer
½

dkernel
ebias
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses"
_tf_keras_layer
½

jkernel
kbias
ltrainable_variables
m	variables
nregularization_losses
o	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"
_tf_keras_layer

piter

qbeta_1

rbeta_2
	sdecay
tlearning_rate$m%m.m/m8m9mBmCmLmMmVmWmdmemjmkm$v%v.v/v8v9vBvCvLvMvVvWvdvevjv kv¡"
	optimizer

$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15"
trackable_list_wrapper

$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
trainable_variables
	variables
ulayer_regularization_losses
regularization_losses
vmetrics
wlayer_metrics

xlayers
ynon_trainable_variables
£__call__
¢_default_save_signature
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
-
Çserving_default"
signature_map
§
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
È__call__
+É&call_and_return_all_conditional_losses"
_tf_keras_layer
©
~trainable_variables
	variables
regularization_losses
	keras_api
Ê__call__
+Ë&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
trainable_variables
	variables
 layer_regularization_losses
regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
¶
	_rng
trainable_variables
	variables
regularization_losses
	keras_api
Ì__call__
+Í&call_and_return_all_conditional_losses"
_tf_keras_layer
¶
	_rng
trainable_variables
	variables
regularization_losses
	keras_api
Î__call__
+Ï&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
 trainable_variables
!	variables
 layer_regularization_losses
"regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_90/kernel
: 2conv2d_90/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
&trainable_variables
'	variables
 layer_regularization_losses
(regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
*trainable_variables
+	variables
 layer_regularization_losses
,regularization_losses
metrics
layer_metrics
layers
non_trainable_variables
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
*:( @2conv2d_91/kernel
:@2conv2d_91/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
0trainable_variables
1	variables
  layer_regularization_losses
2regularization_losses
¡metrics
¢layer_metrics
£layers
¤non_trainable_variables
­__call__
+®&call_and_return_all_conditional_losses
'®"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
4trainable_variables
5	variables
 ¥layer_regularization_losses
6regularization_losses
¦metrics
§layer_metrics
¨layers
©non_trainable_variables
¯__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_92/kernel
:@2conv2d_92/bias
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
:trainable_variables
;	variables
 ªlayer_regularization_losses
<regularization_losses
«metrics
¬layer_metrics
­layers
®non_trainable_variables
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
>trainable_variables
?	variables
 ¯layer_regularization_losses
@regularization_losses
°metrics
±layer_metrics
²layers
³non_trainable_variables
³__call__
+´&call_and_return_all_conditional_losses
'´"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_93/kernel
:@2conv2d_93/bias
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Dtrainable_variables
E	variables
 ´layer_regularization_losses
Fregularization_losses
µmetrics
¶layer_metrics
·layers
¸non_trainable_variables
µ__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Htrainable_variables
I	variables
 ¹layer_regularization_losses
Jregularization_losses
ºmetrics
»layer_metrics
¼layers
½non_trainable_variables
·__call__
+¸&call_and_return_all_conditional_losses
'¸"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_94/kernel
:@2conv2d_94/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ntrainable_variables
O	variables
 ¾layer_regularization_losses
Pregularization_losses
¿metrics
Àlayer_metrics
Álayers
Ânon_trainable_variables
¹__call__
+º&call_and_return_all_conditional_losses
'º"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Rtrainable_variables
S	variables
 Ãlayer_regularization_losses
Tregularization_losses
Ämetrics
Ålayer_metrics
Ælayers
Çnon_trainable_variables
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_95/kernel
:@2conv2d_95/bias
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Xtrainable_variables
Y	variables
 Èlayer_regularization_losses
Zregularization_losses
Émetrics
Êlayer_metrics
Ëlayers
Ìnon_trainable_variables
½__call__
+¾&call_and_return_all_conditional_losses
'¾"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
\trainable_variables
]	variables
 Ílayer_regularization_losses
^regularization_losses
Îmetrics
Ïlayer_metrics
Ðlayers
Ñnon_trainable_variables
¿__call__
+À&call_and_return_all_conditional_losses
'À"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
`trainable_variables
a	variables
 Òlayer_regularization_losses
bregularization_losses
Ómetrics
Ôlayer_metrics
Õlayers
Önon_trainable_variables
Á__call__
+Â&call_and_return_all_conditional_losses
'Â"call_and_return_conditional_losses"
_generic_user_object
": 	@2dense_30/kernel
:@2dense_30/bias
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ftrainable_variables
g	variables
 ×layer_regularization_losses
hregularization_losses
Ømetrics
Ùlayer_metrics
Úlayers
Ûnon_trainable_variables
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
!:@2dense_31/kernel
:2dense_31/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ltrainable_variables
m	variables
 Ülayer_regularization_losses
nregularization_losses
Ýmetrics
Þlayer_metrics
ßlayers
ànon_trainable_variables
Å__call__
+Æ&call_and_return_all_conditional_losses
'Æ"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
0
á0
â1"
trackable_list_wrapper
 "
trackable_dict_wrapper

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ztrainable_variables
{	variables
 ãlayer_regularization_losses
|regularization_losses
ämetrics
ålayer_metrics
ælayers
çnon_trainable_variables
È__call__
+É&call_and_return_all_conditional_losses
'É"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¶
~trainable_variables
	variables
 èlayer_regularization_losses
regularization_losses
émetrics
êlayer_metrics
ëlayers
ìnon_trainable_variables
Ê__call__
+Ë&call_and_return_all_conditional_losses
'Ë"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
/
í
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
	variables
 îlayer_regularization_losses
regularization_losses
ïmetrics
ðlayer_metrics
ñlayers
ònon_trainable_variables
Ì__call__
+Í&call_and_return_all_conditional_losses
'Í"call_and_return_conditional_losses"
_generic_user_object
/
ó
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
trainable_variables
	variables
 ôlayer_regularization_losses
regularization_losses
õmetrics
ölayer_metrics
÷layers
ønon_trainable_variables
Î__call__
+Ï&call_and_return_all_conditional_losses
'Ï"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

ùtotal

úcount
û	variables
ü	keras_api"
_tf_keras_metric
c

ýtotal

þcount
ÿ
_fn_kwargs
	variables
	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:	2Variable
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
ù0
ú1"
trackable_list_wrapper
.
û	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ý0
þ1"
trackable_list_wrapper
.
	variables"
_generic_user_object
/:- 2Adam/conv2d_90/kernel/m
!: 2Adam/conv2d_90/bias/m
/:- @2Adam/conv2d_91/kernel/m
!:@2Adam/conv2d_91/bias/m
/:-@@2Adam/conv2d_92/kernel/m
!:@2Adam/conv2d_92/bias/m
/:-@@2Adam/conv2d_93/kernel/m
!:@2Adam/conv2d_93/bias/m
/:-@@2Adam/conv2d_94/kernel/m
!:@2Adam/conv2d_94/bias/m
/:-@@2Adam/conv2d_95/kernel/m
!:@2Adam/conv2d_95/bias/m
':%	@2Adam/dense_30/kernel/m
 :@2Adam/dense_30/bias/m
&:$@2Adam/dense_31/kernel/m
 :2Adam/dense_31/bias/m
/:- 2Adam/conv2d_90/kernel/v
!: 2Adam/conv2d_90/bias/v
/:- @2Adam/conv2d_91/kernel/v
!:@2Adam/conv2d_91/bias/v
/:-@@2Adam/conv2d_92/kernel/v
!:@2Adam/conv2d_92/bias/v
/:-@@2Adam/conv2d_93/kernel/v
!:@2Adam/conv2d_93/bias/v
/:-@@2Adam/conv2d_94/kernel/v
!:@2Adam/conv2d_94/bias/v
/:-@@2Adam/conv2d_95/kernel/v
!:@2Adam/conv2d_95/bias/v
':%	@2Adam/dense_30/kernel/v
 :@2Adam/dense_30/bias/v
&:$@2Adam/dense_31/kernel/v
 :2Adam/dense_31/bias/v
ÕBÒ
!__inference__wrapped_model_160644sequential_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
.__inference_sequential_17_layer_call_fn_162224
.__inference_sequential_17_layer_call_fn_162261
.__inference_sequential_17_layer_call_fn_162302
.__inference_sequential_17_layer_call_fn_162343À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_17_layer_call_and_return_conditional_losses_162417
I__inference_sequential_17_layer_call_and_return_conditional_losses_162712
I__inference_sequential_17_layer_call_and_return_conditional_losses_162786
I__inference_sequential_17_layer_call_and_return_conditional_losses_163081À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
+__inference_sequential_layer_call_fn_163086
+__inference_sequential_layer_call_fn_163091
+__inference_sequential_layer_call_fn_163096
+__inference_sequential_layer_call_fn_163101
+__inference_sequential_layer_call_fn_163106
+__inference_sequential_layer_call_fn_163111À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ö2ó
F__inference_sequential_layer_call_and_return_conditional_losses_163125
F__inference_sequential_layer_call_and_return_conditional_losses_163139
F__inference_sequential_layer_call_and_return_conditional_losses_163153
F__inference_sequential_layer_call_and_return_conditional_losses_163167
F__inference_sequential_layer_call_and_return_conditional_losses_163177
F__inference_sequential_layer_call_and_return_conditional_losses_163187À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
à2Ý
-__inference_sequential_1_layer_call_fn_163192
-__inference_sequential_1_layer_call_fn_163197
-__inference_sequential_1_layer_call_fn_163206
-__inference_sequential_1_layer_call_fn_163215
-__inference_sequential_1_layer_call_fn_163220
-__inference_sequential_1_layer_call_fn_163229À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2ÿ
H__inference_sequential_1_layer_call_and_return_conditional_losses_163236
H__inference_sequential_1_layer_call_and_return_conditional_losses_163510
H__inference_sequential_1_layer_call_and_return_conditional_losses_163517
H__inference_sequential_1_layer_call_and_return_conditional_losses_163791
H__inference_sequential_1_layer_call_and_return_conditional_losses_163795
H__inference_sequential_1_layer_call_and_return_conditional_losses_164020À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
*__inference_conv2d_90_layer_call_fn_164029¢
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
ï2ì
E__inference_conv2d_90_layer_call_and_return_conditional_losses_164040¢
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
2
1__inference_max_pooling2d_90_layer_call_fn_164045
1__inference_max_pooling2d_90_layer_call_fn_164050¢
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
Ä2Á
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_164055
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_164060¢
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
Ô2Ñ
*__inference_conv2d_91_layer_call_fn_164069¢
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
ï2ì
E__inference_conv2d_91_layer_call_and_return_conditional_losses_164080¢
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
2
1__inference_max_pooling2d_91_layer_call_fn_164085
1__inference_max_pooling2d_91_layer_call_fn_164090¢
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
Ä2Á
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_164095
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_164100¢
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
Ô2Ñ
*__inference_conv2d_92_layer_call_fn_164109¢
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
ï2ì
E__inference_conv2d_92_layer_call_and_return_conditional_losses_164120¢
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
2
1__inference_max_pooling2d_92_layer_call_fn_164125
1__inference_max_pooling2d_92_layer_call_fn_164130¢
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
Ä2Á
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_164135
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_164140¢
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
Ô2Ñ
*__inference_conv2d_93_layer_call_fn_164149¢
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
ï2ì
E__inference_conv2d_93_layer_call_and_return_conditional_losses_164160¢
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
2
1__inference_max_pooling2d_93_layer_call_fn_164165
1__inference_max_pooling2d_93_layer_call_fn_164170¢
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
Ä2Á
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_164175
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_164180¢
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
Ô2Ñ
*__inference_conv2d_94_layer_call_fn_164189¢
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
ï2ì
E__inference_conv2d_94_layer_call_and_return_conditional_losses_164200¢
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
2
1__inference_max_pooling2d_94_layer_call_fn_164205
1__inference_max_pooling2d_94_layer_call_fn_164210¢
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
Ä2Á
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_164215
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_164220¢
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
Ô2Ñ
*__inference_conv2d_95_layer_call_fn_164229¢
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
ï2ì
E__inference_conv2d_95_layer_call_and_return_conditional_losses_164240¢
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
2
1__inference_max_pooling2d_95_layer_call_fn_164245
1__inference_max_pooling2d_95_layer_call_fn_164250¢
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
Ä2Á
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_164255
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_164260¢
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
Õ2Ò
+__inference_flatten_15_layer_call_fn_164265¢
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
ð2í
F__inference_flatten_15_layer_call_and_return_conditional_losses_164271¢
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
Ó2Ð
)__inference_dense_30_layer_call_fn_164280¢
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
î2ë
D__inference_dense_30_layer_call_and_return_conditional_losses_164291¢
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
Ó2Ð
)__inference_dense_31_layer_call_fn_164300¢
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
î2ë
D__inference_dense_31_layer_call_and_return_conditional_losses_164311¢
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
ÔBÑ
$__inference_signature_wrapper_162187sequential_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_rescaling_1_layer_call_fn_164316¢
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
ñ2î
G__inference_rescaling_1_layer_call_and_return_conditional_losses_164324¢
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
Ó2Ð
)__inference_resizing_layer_call_fn_164329¢
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
î2ë
D__inference_resizing_layer_call_and_return_conditional_losses_164338¢
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
2
,__inference_random_flip_layer_call_fn_164343
,__inference_random_flip_layer_call_fn_164350´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_random_flip_layer_call_and_return_conditional_losses_164354
G__inference_random_flip_layer_call_and_return_conditional_losses_164511´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
0__inference_random_rotation_layer_call_fn_164516
0__inference_random_rotation_layer_call_fn_164523´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
K__inference_random_rotation_layer_call_and_return_conditional_losses_164530
K__inference_random_rotation_layer_call_and_return_conditional_losses_164651´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ²
!__inference__wrapped_model_160644$%./89BCLMVWdejkC¢@
9¢6
41
sequential_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_31"
dense_31ÿÿÿÿÿÿÿÿÿ¹
E__inference_conv2d_90_layer_call_and_return_conditional_losses_164040p$%9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿþþ 
 
*__inference_conv2d_90_layer_call_fn_164029c$%9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿþþ µ
E__inference_conv2d_91_layer_call_and_return_conditional_losses_164080l./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ}}@
 
*__inference_conv2d_91_layer_call_fn_164069_./7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ 
ª " ÿÿÿÿÿÿÿÿÿ}}@µ
E__inference_conv2d_92_layer_call_and_return_conditional_losses_164120l897¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ<<@
 
*__inference_conv2d_92_layer_call_fn_164109_897¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ>>@
ª " ÿÿÿÿÿÿÿÿÿ<<@µ
E__inference_conv2d_93_layer_call_and_return_conditional_losses_164160lBC7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_93_layer_call_fn_164149_BC7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@µ
E__inference_conv2d_94_layer_call_and_return_conditional_losses_164200lLM7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_94_layer_call_fn_164189_LM7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@µ
E__inference_conv2d_95_layer_call_and_return_conditional_losses_164240lVW7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_conv2d_95_layer_call_fn_164229_VW7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@¥
D__inference_dense_30_layer_call_and_return_conditional_losses_164291]de0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ@
 }
)__inference_dense_30_layer_call_fn_164280Pde0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ@¤
D__inference_dense_31_layer_call_and_return_conditional_losses_164311\jk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_31_layer_call_fn_164300Ojk/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ«
F__inference_flatten_15_layer_call_and_return_conditional_losses_164271a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_flatten_15_layer_call_fn_164265T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿï
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_164055R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 º
L__inference_max_pooling2d_90_layer_call_and_return_conditional_losses_164060j9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿþþ 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ 
 Ç
1__inference_max_pooling2d_90_layer_call_fn_164045R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_max_pooling2d_90_layer_call_fn_164050]9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿþþ 
ª " ÿÿÿÿÿÿÿÿÿ ï
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_164095R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
L__inference_max_pooling2d_91_layer_call_and_return_conditional_losses_164100h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ}}@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ>>@
 Ç
1__inference_max_pooling2d_91_layer_call_fn_164085R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_max_pooling2d_91_layer_call_fn_164090[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ}}@
ª " ÿÿÿÿÿÿÿÿÿ>>@ï
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_164135R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
L__inference_max_pooling2d_92_layer_call_and_return_conditional_losses_164140h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ<<@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ç
1__inference_max_pooling2d_92_layer_call_fn_164125R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_max_pooling2d_92_layer_call_fn_164130[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ<<@
ª " ÿÿÿÿÿÿÿÿÿ@ï
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_164175R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
L__inference_max_pooling2d_93_layer_call_and_return_conditional_losses_164180h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ç
1__inference_max_pooling2d_93_layer_call_fn_164165R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_max_pooling2d_93_layer_call_fn_164170[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ï
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_164215R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
L__inference_max_pooling2d_94_layer_call_and_return_conditional_losses_164220h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ç
1__inference_max_pooling2d_94_layer_call_fn_164205R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_max_pooling2d_94_layer_call_fn_164210[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@ï
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_164255R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¸
L__inference_max_pooling2d_95_layer_call_and_return_conditional_losses_164260h7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Ç
1__inference_max_pooling2d_95_layer_call_fn_164245R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
1__inference_max_pooling2d_95_layer_call_fn_164250[7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª " ÿÿÿÿÿÿÿÿÿ@±
G__inference_random_flip_layer_call_and_return_conditional_losses_164354f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 µ
G__inference_random_flip_layer_call_and_return_conditional_losses_164511jí8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
,__inference_random_flip_layer_call_fn_164343Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_random_flip_layer_call_fn_164350]í8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿµ
K__inference_random_rotation_layer_call_and_return_conditional_losses_164530f8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ¹
K__inference_random_rotation_layer_call_and_return_conditional_losses_164651jó8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
0__inference_random_rotation_layer_call_fn_164516Y8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
0__inference_random_rotation_layer_call_fn_164523]ó8¢5
.¢+
%"
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ­
G__inference_rescaling_1_layer_call_and_return_conditional_losses_164324b4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿØ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿØ
 
,__inference_rescaling_1_layer_call_fn_164316U4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿØ
ª "ÿÿÿÿÿÿÿÿÿØ¢
D__inference_resizing_layer_call_and_return_conditional_losses_164338Z4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿØ
ª ""¢

0
 z
)__inference_resizing_layer_call_fn_164329M4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿØ
ª "É
I__inference_sequential_17_layer_call_and_return_conditional_losses_162417|$%./89BCLMVWdejkA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Î
I__inference_sequential_17_layer_call_and_return_conditional_losses_162712íó$%./89BCLMVWdejkA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ô
I__inference_sequential_17_layer_call_and_return_conditional_losses_162786$%./89BCLMVWdejkK¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ø
I__inference_sequential_17_layer_call_and_return_conditional_losses_163081íó$%./89BCLMVWdejkK¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 «
.__inference_sequential_17_layer_call_fn_162224y$%./89BCLMVWdejkK¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
.__inference_sequential_17_layer_call_fn_162261o$%./89BCLMVWdejkA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¥
.__inference_sequential_17_layer_call_fn_162302síó$%./89BCLMVWdejkA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¯
.__inference_sequential_17_layer_call_fn_162343}íó$%./89BCLMVWdejkK¢H
A¢>
41
sequential_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¶
H__inference_sequential_1_layer_call_and_return_conditional_losses_163236j<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 ¼
H__inference_sequential_1_layer_call_and_return_conditional_losses_163510píó<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 Á
H__inference_sequential_1_layer_call_and_return_conditional_losses_163517uG¢D
=¢:
0-
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 Ç
H__inference_sequential_1_layer_call_and_return_conditional_losses_163791{íóG¢D
=¢:
0-
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 À
H__inference_sequential_1_layer_call_and_return_conditional_losses_163795tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Æ
H__inference_sequential_1_layer_call_and_return_conditional_losses_164020zíóA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_1_layer_call_fn_163192hG¢D
=¢:
0-
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_163197]<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_163206cíó<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_163215níóG¢D
=¢:
0-
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_163220gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_1_layer_call_fn_163229míóA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿ¬
F__inference_sequential_layer_call_and_return_conditional_losses_163125b<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿØ
p 

 
ª ""¢

0
 ¬
F__inference_sequential_layer_call_and_return_conditional_losses_163139b<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿØ
p

 
ª ""¢

0
 ·
F__inference_sequential_layer_call_and_return_conditional_losses_163153mG¢D
=¢:
0-
rescaling_1_inputÿÿÿÿÿÿÿÿÿØ
p 

 
ª ""¢

0
 ·
F__inference_sequential_layer_call_and_return_conditional_losses_163167mG¢D
=¢:
0-
rescaling_1_inputÿÿÿÿÿÿÿÿÿØ
p

 
ª ""¢

0
 ¾
F__inference_sequential_layer_call_and_return_conditional_losses_163177tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¾
F__inference_sequential_layer_call_and_return_conditional_losses_163187tA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_sequential_layer_call_fn_163086`G¢D
=¢:
0-
rescaling_1_inputÿÿÿÿÿÿÿÿÿØ
p 

 
ª "
+__inference_sequential_layer_call_fn_163091U<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿØ
p 

 
ª "
+__inference_sequential_layer_call_fn_163096U<¢9
2¢/
%"
inputsÿÿÿÿÿÿÿÿÿØ
p

 
ª "
+__inference_sequential_layer_call_fn_163101`G¢D
=¢:
0-
rescaling_1_inputÿÿÿÿÿÿÿÿÿØ
p

 
ª "
+__inference_sequential_layer_call_fn_163106gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ""ÿÿÿÿÿÿÿÿÿ
+__inference_sequential_layer_call_fn_163111gA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ""ÿÿÿÿÿÿÿÿÿÉ
$__inference_signature_wrapper_162187 $%./89BCLMVWdejkW¢T
¢ 
MªJ
H
sequential_input41
sequential_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_31"
dense_31ÿÿÿÿÿÿÿÿÿ