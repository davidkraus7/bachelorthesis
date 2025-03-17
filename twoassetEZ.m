%% Partial equilibrium of two-asset model with recursive preferences

clear all
close all
clc
tic;

% We specify the model parameters

rho = 0.05;
gamma = 4;
psi = 0.5;
sigma = 1/psi;
mu = 0.07;
r = 0.03;
w = 1;
nu = 0.2;
nu2 = nu^2;
phi = 0;
a_min = -phi;
a_max = 1000;
z = [0.03, 0.09];
lambda = [0.5, 0.5];

% We create the asset grid

I = 20000;
a = linspace(a_min, a_max, I)';
da = a(2) - a(1);
aa = [a, a];

% We can already construct the transition matrix for income

Aswitch = [-speye(I)*lambda(1),speye(I)*lambda(1);speye(I)*lambda(2),-speye(I)*lambda(2)];

% We create empty matrices for finite differences

dVf = zeros(I,2);             
dVb = zeros(I,2);
dV2f = zeros(I,2);
dV2b = zeros(I,2);

% Value of staying put with only safe assets as initial guess for v

v0 = 1/rho * (r*aa + w*z).^(1-gamma)/(1-gamma);

% We iteratively solve the HJB

v = v0;
max_iter = 500;
step = 100;
crit = 1e-3;

for i = 1:max_iter
    
    V = v;
    
    % We compute forward and backward finite difference v'

    dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
    tlim = (mu-r)/(gamma*nu^2);
    dVf(I,:) = rho*((1-gamma)*V(I,:)).^((sigma-gamma)/(1-gamma)).*(w*z + r*a(end) + tlim*a(end)*(mu-r)).^(-sigma);
    
    dVb(2:I,:) = (V(2:I,:) - V(1:I-1,:))/da;
    dVb(1,:) = rho*((1-gamma)*V(1,:)).^((sigma-gamma)/(1-gamma)).*(r*a(1)+w*z).^(-sigma);

    % We compute central difference v''
    % We impose upper boundary condition as described in the notes
    
    dV2f(2:I-1,:) = (V(3:I,:)-2*V(2:I-1,:)+V(1:I-2,:))/da^2;
    dV2f(1,:) = (dVf(2,:)-dVf(1,:))/da;
    dV2f(I,:) = -gamma*dVf(I,:)/a(end);
    
    dV2b(2:I-1,:) = (V(3:I,:)-2*V(2:I-1,:)+V(1:I-2,:))/da^2;
    dV2b(1,:) = (dVb(2,:)-dVb(1,:))/da;
    dV2b(I,:) = -gamma*dVb(I,:)/a(end);

    % We compute implied consumption, portfolio choice and savings

    cf = rho^psi*((1-gamma)*V).^((1-psi*gamma)/(1-gamma)).*dVf.^(-psi);
    tf = max(-dVf./dV2f.*(mu-r)./(nu^2*aa), 0);
    tf = min(tf, 1);
    sf = w*z + r*aa +tf.*aa*(mu - r) - cf;

    cb = rho^psi*((1-gamma)*V).^((1-psi*gamma)/(1-gamma)).*dVb.^(-psi);
    tb = max(-dVb./dV2b.*(mu-r)./(nu^2*aa), 0);
    tb = min(tb, 1);
    sb = w*z + r*aa +tb.*aa*(mu - r) - cb;

    t0 = (tf + tb)/2;
    c0 = w*z + r*aa + t0.*aa*(mu-r);
    dV0 = rho*((1-gamma)*V).^((sigma-gamma)/(1-gamma)).*((c0).^(-sigma));

    % We construct the upwind scheme approximation of v'

    If = sf > 1e-10;
    Ib = sb < -1e-10;
    I0 = 1-If-Ib;

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;

    % We store policy functions and flow utility

    c = rho^psi*((1-gamma)*V).^((1-psi*gamma)/(1-gamma)).*dV_Upwind.^(-psi);
    t = max(-dV_Upwind./dV2b.*(mu-r)./(nu^2*aa), 0);
    t = min(t, 1);
    t(1,:) = [1,1];
    s = w*z + r*aa + t.*aa*(mu - r) - c;

    % We construct the transition matrix
    
    % Upwind scheme for v'
    X = - min(sb,0)/da;
    Y = - max(sf,0)/da + min(sb,0)/da;
    Z = max(sf,0)/da;
    
    % Central difference for v''
    X = X + 0.5*nu^2.*aa.^2.*t.^2./(da^2);
    Z = Z + 0.5*nu^2.*aa.^2.*t.^2./(da^2);
    Y = Y - nu^2.*aa.^2.*t.^2./(da^2);
    
    % We impose upper boundary condition as described in
    % the notes
    xi = -0.5*a_max*(mu-r)^2/gamma/nu2;
    X(I,:) = -min(sb(I,:),0)/da - xi/da;
    Y(I,:) = -max(sf(I,:),0)/da + min(sb(I,:),0)/da + xi/da;
    Z(I,:) = max(sf(I,:),0)/da;
    
    updiag=[0];
    for j=1:2
        updiag=[updiag;Z(1:I-1,j);0];
    end
        
    centdiag=reshape(Y,2*I,1);
        
    lowdiag=X(2:I,1);
    lowdiag=[lowdiag;0;X(2:I,2)];

    A=spdiags(centdiag,0,2*I,2*I)+spdiags([updiag;0],1,2*I,2*I)+spdiags([lowdiag;0],-1,2*I,2*I);
    A = A + Aswitch;

    if max(abs(sum(A,2)))>10^(-9)
       disp('Improper Transition Matrix')
       break
    end
    
    % We update the value function using an implicit scheme

    b = rho/(1-sigma)*c.^(1-sigma).*((1-gamma)*V).^((sigma-gamma)/(1-gamma)) + 1/step.*V;
    b_stacked = [b(:,1);b(:,2)];
    B = (1/step + rho*(1-gamma)/(1-sigma)).*speye(2*I) - A;

    V_stacked = B\b_stacked;
    V = [V_stacked(1:I),V_stacked(I+1:2*I)];
    
    Vchange = V - v;
    v = V;

    dist = max(max(abs(Vchange)));
    if dist<crit
        disp('Value Function Converged, Iteration = ')
        disp(i)
        break
    end

    % We use a relaxtion method to avoid numerical instability

    relax = 0.7;
    V = relax*V + (1-relax)*v;
    v = V;
end

% We reconstruct the transition matrix with a reflecting barrier

% Upwind scheme for v'
X = - min(sb,0)/da;
Z = max(sf,0)/da;
Y = - X - Z;
    
% Central difference for v''
X = X + 0.5*nu^2.*aa.^2.*t.^2./(da^2);
Z = Z + 0.5*nu^2.*aa.^2.*t.^2./(da^2);
Y = Y - nu^2.*aa.^2.*t.^2./(da^2);

% Construct transition matrix
updiag=[0];
for j=1:2
    updiag=[updiag;Z(1:I-1,j);0];
end
    
centdiag=reshape(Y,2*I,1);
    
lowdiag=X(2:I,1);
lowdiag=[lowdiag;0;X(2:I,2)];

A=spdiags(centdiag,0,2*I,2*I)+spdiags([updiag;0],1,2*I,2*I)+spdiags([lowdiag;0],-1,2*I,2*I);

% Reflecting barrier at upper boundary of the state space

A(I, I) = Y(I, 1) + Z(I, 1);
A(2*I, 2*I) = Y(I, 2) + Z(I, 2);

A = A + Aswitch;

% We solve for the stationary distribution

tA = A';

e = zeros(2*I,1);
i_fix = 1;
e(i_fix)=.1;
row = [zeros(1,i_fix-1),1,zeros(1,2*I-i_fix)];
tA(i_fix,:) = row;

g_stacked = tA\e;
g_sum = g_stacked'*ones(2*I,1)*da;
g_stacked = g_stacked./g_sum;

gg = reshape(g_stacked,I,2);

toc;

amax1 = 2;
amin1 = a_min-0.05*amax1;
figure(1)
h1 = plot(a,c(:,1),'r',a,c(:,2),'b','LineWidth',2);
set(gca,'FontSize',16)
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Consumption','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$c_{L}(a)$', '$c_{H}(a)$'}, ...
       'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 30)

amax1 = 3;
amin1 = a_min-0.05*amax1;
figure(1)
h1 = plot(a,s(:,1),'r',a,s(:,2),'b','LineWidth',2);
set(gca,'FontSize',16)
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Savings','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
yline(0, 'k--', 'LineWidth', 1.5); % Black dashed line at y=0
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$s_{L}(a)$', '$s_{H}(a)$'}, ...
       'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 30)

amax1 = 3;
amin1 = a_min-0.05*amax1;
figure(1)
h1 = plot(a,t(:,1),'r',a,t(:,2),'b','LineWidth',2);
set(gca,'FontSize',16)
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Risky asset share','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$\theta_{L}(a)$', '$\theta_{H}(a)$'}, ...
       'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 30)

amax1 = 5;
amin1 = a_min-0.05*amax1;
figure(1)
h1 = plot(a,gg(:,1),'r',a,gg(:,2),'b','LineWidth',2);
set(gca,'FontSize',16)
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Density','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$g_{L}(a)$', '$g_{H}(a)$'}, ...
       'Location', 'NorthEast', 'Interpreter', 'latex', 'FontSize', 30)
