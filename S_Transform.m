%Quantum S_Transform generalization: Third version (14/02/2022)
%Select: Signal length, Decomposition levels and Bit_precision
%The signal is randomly generated.

close all;
clear;
clc;
qlib

%% Initialization

%Length of the signal
Signal_Len = input('Enter the signal length: '); 

%Decomposition level
%The maximun decomposition level is: Signal_Len = 2^(max(D_level) + 1)
%Therefore, D_level <= max(D_level)
D_level = input('Enter the decomposition level: ');

%Bit precision of the signal
Bit_precision = input('Enter the bit precision: ');
Bit_precision = Bit_precision + 1;

%Random signal generation (Integer signal)
xmin = 0;
xmax = 2^Bit_precision;
Signal = fix(xmin + rand(1,Signal_Len)*(xmax - xmin));

% Signal = [-1 -2 2 1];

disp('Singal')
disp(Signal)

for d = 1:D_level
      
%Signal Lenght (base 2)
S_length = log(size(Signal,2)/2)/log(2); 

%Number of Positions qubits
Number_position = S_length;

%Number of auxiliary qubits for add
aux_a = Bit_precision +1;

%Number of auxiliary qubits for sub
aux_s = Bit_precision;

%Number of odd elements
q_odd = zeros(1, 2^(Bit_precision))'; % 2^n 
q_odd(1) = 1;

%Number of even elements
q_even = q_odd;

%auxiliary vector (addition operation)
av_a = zeros(1,2^(aux_a))';
av_a(1) = 1;

%auxiliary vector (subtraction operation)
av_s = zeros(1,2^(aux_s))';
av_s(1) = 1;

%Position vector
x_1 = zeros(1, 2)';
x_1(1) = 1;

x_1 = act_on(pure2dm(x_1), QLib.gates.Hadamard, 1);
x_1 = dm2pure(x_1);

x = x_1;

%spmd

for k = 1:(Number_position-1)
    x = kron(x_1, x);
end

%Storing the signal elements 
x = kron(x, q_odd);
x = kron(x, q_even);

x(:,:)=0;

for k=1:2^Number_position
    j=k-1;
    Image_element_odd  = dec2bin( Signal(2*k-1), Bit_precision);
    if Signal(2*k-1)<0
        Image_element_odd  = dec2bin( Signal(2*k-1), 8);
        Image_element_odd = Image_element_odd(end-Bit_precision+1:end);
    end
    
    Image_element_even = dec2bin( Signal(2*k),   Bit_precision);
    if Signal(2*k)<0
        Image_element_even = dec2bin( Signal(2*k), 8);
        Image_element_even = Image_element_even(end-Bit_precision+1:end);
    end
    position_bin = dec2bin(j, Number_position);
    
    Position_number = bin2dec(strcat(position_bin, Image_element_odd, Image_element_even))+1;
    x(Position_number) = sqrt(1/(2^Number_position));
end


%% Controlled-V and V' gates
%Generate the operators
V = ((1i+1)/2).*[1 -1i; -1i  1]; %V operator
VP = V';

CV = control_gate(V);
CVP = control_gate(VP);

x_s = x; %Copy the states for the subtraction operation

%% Addition 
%q_odd initial position (q0): |qn...q0>
P_odd = Number_position + Bit_precision;

%q_even initial position (q0): |qn...q0>
P_even = P_odd + Bit_precision;

%Aux qubit initial position 
P_aux = P_even + aux_a;

%Tensor product 
x = kron(x, av_a);

%Addition operation
for i=1:Bit_precision
    j = i-1;
    x = act_on_pure(x, CVP, [P_even-j P_aux-i]);
    x = act_on_pure(x, CVP, [P_odd-j  P_aux-i]);
    x = act_on_pure(x, CVP, [P_aux-j  P_aux-i]);
    
    x = act_on_pure(x, QLib.gates.CNOT, [P_odd-j  P_even-j]);
    x = act_on_pure(x, QLib.gates.CNOT, [P_even-j P_aux-j] );
    x = act_on_pure(x, QLib.gates.CNOT, [P_odd-j  P_even-j]);
    
    x = act_on_pure(x, CV, [P_aux-j P_aux-i]);
end

% States = find(x~=0);
% States = States - 1;
% States = dec2bin(States);
% disp('Sum')
% disp(bin2dec(States(:,P_even+1:P_even+aux_a))')

%% Rounding

%Rouding variable
R_v = zeros(1,2)';
R_v(1) = 1;

%Rounding variable position 
P_Rv = P_aux + 1;

%Tensor product 
x = kron(x, R_v);

%Swap operation
x = act_on_pure(x, QLib.gates.CNOT, [P_aux P_Rv] );
x = act_on_pure(x, QLib.gates.CNOT, [P_Rv  P_aux]);
x = act_on_pure(x, QLib.gates.CNOT, [P_aux P_Rv] );

% States = find(x~=0);
% States = States - 1;
% States = dec2bin(States);
% disp('Sub(1): odd')
% disp(bin2dec(States(:,P_even+1:P_even+aux_a))')

%Halving operation
for i=1:aux_a-1
    j=i-1;
    x = act_on_pure(x, QLib.gates.CNOT, [P_aux-j  P_aux-i]);
    x = act_on_pure(x, QLib.gates.CNOT, [P_aux-i  P_aux-j]);
    x = act_on_pure(x, QLib.gates.CNOT, [P_aux-j  P_aux-i]);
end

States = find(x~=0);
States = States - 1;
States = dec2bin(States);

A = bin2dec(States(:,P_even+1:P_even+aux_a))';

%% Subtraction

%Aux qubit initial position 
P_aux_s = P_even + aux_s;

%Tensor product 
x_s = kron(x_s, av_s);

%Subtraction operation
x_s = act_on_pure(x_s, CVP,             [P_even P_aux_s]);
x_s = act_on_pure(x_s, QLib.gates.CNOT, [P_odd P_even]  );
x_s = act_on_pure(x_s, CV,              [P_odd  P_aux_s]);
x_s = act_on_pure(x_s, CV,              [P_even P_aux_s]);

for i=1:Bit_precision-1
    x_s = act_on_pure(x_s, CVP,             [P_even-i P_aux_s-i]);
    x_s = act_on_pure(x_s, QLib.gates.CNOT, [P_odd-i P_even-i] );
    x_s = act_on_pure(x_s, CV,              [P_odd-i  P_aux_s-i]);
    
    x_s = act_on_pure(x_s, QLib.gates.CNOT, [P_aux_s-i+1 P_even-i] );
    x_s = act_on_pure(x_s, CV,              [P_aux_s-i+1 P_aux_s-i]);
    x_s = act_on_pure(x_s, CV,              [P_even-i    P_aux_s-i]);
end

%diff = Odd-even
States = find(x_s~=0);
States = States - 1;
States = dec2bin(States);

states_r = States(:,P_odd+1:P_even); %Extract the element stored in the even components
diff = strcat( States(:,P_even+1:P_even+1), states_r); %Concatenate the Borrow bit with
%The previous components
diff = bin2dec(diff);

%To transform the binary negative numbers
for k=1:2^(Bit_precision)-1
    Minus_number = dec2bin(typecast(int8(-k),'uint8'));
    Minu_number = Minus_number(1,end-Bit_precision:end);
    diff(diff==bin2dec(Minu_number)) = -k;
end

%disp('|D>')
%disp('|D_'+(d-1)+' >')
disp(strcat('|D_',int2str(d-1),' >'))
disp(diff')

if d == D_level
    disp('|A >')
    disp(A)
end

Signal = A;
end

