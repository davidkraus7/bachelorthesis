%% Partial Equilibrium of two-asset model with time-varying equity premium and recursive preferences

clear all
close all
clc
tic;

% We specify the model parameters

rho = 0.05;
gamma = 4;
psi = 0.5;
sigma = 1/psi;
mu = [0.06, 0.08];
muu = [mu(1), mu(1), mu(2), mu(2)];
lambda_mu = [0.5, 0.5];
r = 0.03;
w = 1;
nu = 0.2;
nu2 = nu^2;
phi = 0;
a_min = -phi;
a_max = 1000;
z1 = [0.03, 0.09];
lambda_z1 = [0.5, 0.5];
z2 = [0.03, 0.09];
lambda_z2 = [0.5, 0.5];

% We create the asset grid

I = 20000;
a = linspace(a_min, a_max, I)';
da = a(2) - a(1);
aa = [a, a, a, a];

% We can already construct the transition matrix for idiosyncratic labour income

Z1switch = [-speye(I)*lambda_z1(1),speye(I)*lambda_z1(1);speye(I)*lambda_z1(2),-speye(I)*lambda_z1(2)];
Z2switch = [-speye(I)*lambda_z2(1),speye(I)*lambda_z2(1);speye(I)*lambda_z2(2),-speye(I)*lambda_z2(2)];
Zswitch = [Z1switch, zeros(2*I); zeros(2*I), Z2switch];

% We can similarly construct the transition matrix for the idiosyncratic equity premium

Mswitch = [-speye(2*I)*lambda_mu(1), speye(2*I)*lambda_mu(1); speye(2*I)*lambda_mu(2), -speye(2*I)*lambda_mu(2)];

% We create empty matrices for finite differences

dVf = zeros(I,4);             
dVb = zeros(I,4);
dV2f = zeros(I,4);
dV2b = zeros(I,4);

% Value of staying put with only safe assets as initial guess for v

zz = [z1, z2];
v0 = 1/rho * (r*aa + w*zz).^(1-gamma)/(1-gamma);

% We iteratively solve the HJB

v = v0;
max_iter = 200;
step = 100;
crit = 1e-3;

for i = 1:max_iter
    
    V = v;
    
    % We compute forward and backward finite difference v'

    dVf(1:I-1,:) = (V(2:I,:)-V(1:I-1,:))/da;
    tlim = (muu-r)/(gamma*nu^2);
    dVf(I,:) = rho*((1-gamma)*V(I,:)).^((sigma-gamma)/(1-gamma)).*(w*[z1,z2] + r*a(end) + tlim*a(end).*(muu-r)).^(-sigma);
    
    dVb(2:I,:) = (V(2:I,:) - V(1:I-1,:))/da;
    dVb(1,:) = rho*((1-gamma)*V(1,:)).^((sigma-gamma)/(1-gamma)).*(r*a(1)+w*[z1,z2]).^(-sigma);

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
    tf = max(-dVf./dV2f.*(muu-r)./(nu^2*aa), 0);
    tf = min(tf, 1);
    sf = w*zz + r*aa +tf.*aa.*(muu - r) - cf;

    cb = rho^psi*((1-gamma)*V).^((1-psi*gamma)/(1-gamma)).*dVb.^(-psi);
    tb = max(-dVb./dV2b.*(muu-r)./(nu^2*aa), 0);
    tb = min(tb, 1);
    sb = w*zz + r*aa +tb.*aa.*(muu - r) - cb;

    t0 = (tf + tb)/2;
    c0 = w*zz + r*aa + t0.*aa.*(muu-r);
    dV0 = rho*((1-gamma)*V).^((sigma-gamma)/(1-gamma)).*((c0).^(-sigma));

    % We construct the upwind scheme approximation of v'

    If = sf > 1e-10;
    Ib = sb < -1e-10;
    I0 = 1-If-Ib;

    dV_Upwind = dVf.*If + dVb.*Ib + dV0.*I0;

    % We store policy functions and the flow payoff

    c = rho^psi*((1-gamma)*V).^((1-psi*gamma)/(1-gamma)).*dV_Upwind.^(-psi);
    t = max(-dV_Upwind./dV2b.*(muu-r)./(nu^2*aa), 0);
    t = min(t, 1);
    t(1,:) = [1,1,1,1];
    s = w*zz + r*aa + t.*aa.*(muu - r) - c;

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
    xi = -0.5*a_max*(muu-r).^2/gamma/nu2;
    X(I,:) = -min(sb(I,:),0)/da - xi/da;
    Y(I,:) = -max(sf(I,:),0)/da + min(sb(I,:),0)/da + xi/da;
    Z(I,:) = max(sf(I,:),0)/da;
    
    lowdiag=[X(2:I,1)];
    for j=2:4
        lowdiag=[lowdiag;0;X(2:I,j)];
    end

    centdiag=reshape(Y,4*I,1);

    updiag=[Z(1:I-1,1)];
    for j=2:4
        updiag=[updiag;0;Z(1:I-1,j)];
    end
        
    D = [ [lowdiag; 0], centdiag, [0; updiag] ];  
    d = [-1, 0, 1];
    
    A = spdiags(D, d, 4*I, 4*I);
    A = A + Zswitch + Mswitch;

    if max(abs(sum(A,2)))>10^(-9)
       disp('Improper Transition Matrix')
       break
    end
    
    % We update the value function using a semi-implicit scheme

    b = rho/(1-sigma)*c.^(1-sigma).*((1-gamma)*V).^((sigma-gamma)/(1-gamma)) + 1/step.*V;
    b_stacked = [b(:,1);b(:,2); b(:,3); b(:,4)];
    B = (1/step + rho*(1-gamma)/(1-sigma)).*speye(4*I) - A;

    V_stacked = B\b_stacked;
    V = [V_stacked(1:I),V_stacked(I+1:2*I), V_stacked(2*I+1:3*I), V_stacked(3*I+1:4*I)];
    
    Vchange = V - v;
    v = V;

    dist = max(max(abs(Vchange)));
    if dist<crit
        disp('Value Function Converged, Iteration = ')
        disp(i)
        break
    end
end

% We reconstruct the transition matrix with a reflecting barrier

% Upwind scheme for v'
X = - min(sb,0)/da;
Y = - max(sf,0)/da + min(sb,0)/da;
Z = max(sf,0)/da;
    
% Central difference for v''
X = X + 0.5*nu^2.*aa.^2.*t.^2./(da^2);
Z = Z + 0.5*nu^2.*aa.^2.*t.^2./(da^2);
Y = Y - nu^2.*aa.^2.*t.^2./(da^2);

% Construct transition matrix
lowdiag=[X(2:I,1)];
for j=2:4
    lowdiag=[lowdiag;0;X(2:I,j)];
end

centdiag=reshape(Y,4*I,1);

updiag=[Z(1:I-1,1)];
for j=2:4
    updiag=[updiag;0;Z(1:I-1,j)];
end
    
D = [ [lowdiag; 0], centdiag, [0; updiag] ];  
d = [-1, 0, 1];

A = spdiags(D, d, 4*I, 4*I);

% Reflecting barrier at upper boundary of the state space
A(I, I) = Y(I, 1) + Z(I, 1);
A(2*I, 2*I) = Y(I, 2) + Z(I, 2);
A(3*I, 3*I) = Y(I, 3) + Z(I, 3);
A(4*I, 4*I) = Y(I, 4) + Z(I, 4);

A = A + Zswitch + Mswitch;

% We solve for the stationary distribution

tA = A';

e = zeros(4*I,1);
i_fix = 1;
e(i_fix)=.1;
row = [zeros(1,i_fix-1),1,zeros(1,4*I-i_fix)];
tA(i_fix,:) = row;

g_stacked = tA\e;
g_sum = g_stacked'*ones(4*I,1)*da;
g_stacked = g_stacked./g_sum;

gg = reshape(g_stacked,I,4);

toc;

amax1 = 2;
amin1 = a_min - 0.05*amax1;
figure(1)
h1 = plot(a,c(:,1),'r--',a,c(:,2),'b--',a,c(:,3),'r',a,c(:,4), 'LineWidth',2);
set(gca,'FontSize',16)
text(-0.555,-.085,'$\underline{a}$','FontSize',16,'interpreter','latex')
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Consumption','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$c_{L,\mu_-}$', '$c_{H,\mu_-}$', '$c_{L,\mu_+}$', '$c_{H,\mu_+}$',}, ...
       'Location', 'SouthEast', 'Interpreter', 'latex', 'FontSize', 30)

amax1 = 3;
amin1 = a_min - 0.05*amax1;
figure(1)
h1 = plot(a,s(:,1),'r--',a,s(:,2),'b--',a,s(:,3),'r',a,s(:,4),'b','LineWidth',2);
set(gca,'FontSize',16)
text(-0.555,-.085,'$\underline{a}$','FontSize',16,'interpreter','latex')
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Savings','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
yline(0, 'k--', 'LineWidth', 1); % Black dashed line at y=0
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$s_{L,\mu_-}$', '$s_{H,\mu_-}$', '$s_{L,\mu_+}$', '$s_{H,\mu_+}$',}, ...
       'Location', 'SouthEast', 'Interpreter', 'latex', 'FontSize', 30)

amax1 = 3;
amin1 = a_min - 0.05*amax1;
figure(1)
h1 = plot(a,t(:,1),'r--',a,t(:,2),'b--',a,t(:,3),'r',a,t(:,4),'b', 'LineWidth',2);
set(gca,'FontSize',16)
text(-0.555,-.085,'$\underline{a}$','FontSize',16,'interpreter','latex')
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Risky asset share','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$\theta_{L,\mu_-}$', '$\theta_{H,\mu_-}$', '$\theta_{L,\mu_+}$', '$\theta_{H,\mu_+}$',}, ...
       'Location', 'SouthEast', 'Interpreter', 'latex', 'FontSize', 30)

amax1 = 5;
amin1 = a_min - 0.05*amax1;
figure(1)
h1 = plot(a,gg(:,1),'r--',a,gg(:,2),'b--',a,gg(:,3),'r',a,gg(:,4),'b','LineWidth',2);
set(gca,'FontSize',16)
text(-0.555,-.085,'$\underline{a}$','FontSize',16,'interpreter','latex')
xlabel('Wealth','interpreter','latex', 'FontSize', 35)
ylabel('Density','interpreter','latex', 'FontSize', 35)
xlim([amin1 amax1])
line([-phi -phi], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1)
set(gca, 'XTick', []);
set(gca, 'YTick', []);
legend(h1, {'$g_{L,\mu_-}$', '$g_{H,\mu_-}$', '$g_{L,\mu_+}$', '$g_{H,\mu_+}$',}, ...
       'Location', 'SouthEast', 'Interpreter', 'latex', 'FontSize', 30)