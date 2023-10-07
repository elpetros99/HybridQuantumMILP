function [v_info]=makebdc_custom()
    cases =  {'case9_DCOPF'};
    mpc = eval(cases{1});
    [Bbus, Bline, ~, ~] = makeBdc(mpc.baseMVA, mpc.bus, mpc.branch);
    v_info.Bbus = full(Bbus);
    v_info.Bline = full(Bline);
    c=1;
    iter=1;
    path_input = strcat('.\Trained_Neural_Networks\',cases{c},'\',num2str(iter),'\');
    delta=0;
    nr_neurons=20;
    % v_info.Bbus=Bbus;
    % v_info.Bline=Bline;
%     save("./test")
    
% define named indices into data matrices
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, ...
    VA, BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[PW_LINEAR, POLYNOMIAL, MODEL, STARTUP, SHUTDOWN, NCOST, COST] = idx_cost;


nb = size(mpc.bus,1);
v_info.nb=nb;
ng = size(mpc.gen,1);
v_info.ng=ng;
nline = size(mpc.branch,1);
v_info.nline=nline;
% identify the loads which are non-zero
ID_loads = find(mpc.bus(:,PD)~=0);
v_info.ID_loads=ID_loads;
nd=size(ID_loads,1);
v_info.nd=nd;
pd_max = mpc.bus(ID_loads,PD);
v_info.pd_max=pd_max;
%map from buses to loads
M_d = zeros(nb,nd);
for i = 1:nd
    M_d(ID_loads(i),i) = 1;
end
v_info.M_d=M_d;
%map from generators to buses
M_g = zeros(nb,ng);
ID_gen = mpc.gen(:,GEN_BUS);
for i = 1:ng
    M_g(ID_gen(i),i) = 1;
end
v_info.M_g=M_g;

% Here we assume that the loading ranges from 60% to 100%
pd_min =  pd_max.*0.6;
v_info.pd_min=pd_min;
pd_delta = pd_max.*0.4;
v_info.pd_delta=pd_delta;
pg_delta = mpc.gen(2:end,PMAX)-mpc.gen(2:end,PMIN);
v_info.pg_delta=pg_delta;
% options

% Load the neural network weights and biases
W_input = csvread(strcat(path_input,'W0.csv')).';
v_info.W_input=W_input;
W_output = csvread(strcat(path_input,'W2.csv')).'; % not clear how the indexing works here (going from layer 1 to layer 2)
v_info.W_output=W_output;
W{1} = csvread(strcat(path_input,'W1.csv')).';
v_info.W=W;
% W{2} = csvread(strcat(path_input,'W2.csv')).';
bias{1} = csvread(strcat(path_input,'b0.csv')); %net.b; % bias
bias{2} = csvread(strcat(path_input,'b1.csv'));
bias{3} = csvread(strcat(path_input,'b2.csv'));
v_info.bias=bias;
% bias{4} = csvread(strcat(path_input,'b3.csv'));

Input_NN = csvread(strcat(path_input,'features_test.csv'));
v_info.Input_NN=Input_NN;
Output_NN = csvread(strcat(path_input,'labels_test.csv'));
v_info.Output_NN=Output_NN;
% load tightened ReLU bounds
load(strcat(path_input,'zk_hat_min'));
load(strcat(path_input,'zk_hat_max'));
v_info.zk_hat_min=zk_hat_min;
v_info.zk_hat_max=zk_hat_max;
% load Relu stability (active/inactive ReLUs)
load(strcat(path_input,'ReLU_stability_inactive'));
load(strcat(path_input,'ReLU_stability_active'));
v_info.ReLU_stability_inactive=ReLU_stability_inactive;
v_info.ReLU_stability_active=ReLU_stability_active;
ReLU_layers = 2;
v_info.ReLU_layers=ReLU_layers;
PGMAX=mpc.gen(:,PMAX);
v_info.PGMAX=PGMAX;

end 