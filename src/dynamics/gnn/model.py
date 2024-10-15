import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        )
        self.output_size = output_size

    def forward(self, x):
        s = x.size()
        print(f"x input size: {s}")
        x = self.model(x.view(-1, s[-1]))
        return x.view(list(s[:-1]) + [self.output_size])

class Propagator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Propagator, self).__init__()

        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x, res=None):
        s_x = x.size()

        x = self.linear(x.view(-1, s_x[-1]))

        if res is not None:
            s_res = res.size()
            x += res.view(-1, s_res[-1])

        x = self.relu(x).view(list(s_x[:-1]) + [self.output_size])
        return x

class ParticlePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ParticlePredictor, self).__init__()

        self.linear_0 = nn.Linear(input_size, hidden_size)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.output_size = output_size

    def forward(self, x):
        s_x = x.size()

        x = x.view(-1, s_x[-1])
        x = self.relu(self.linear_0(x))
        x = self.relu(self.linear_1(x))

        return self.linear_2(x).view(list(s_x[:-1]) + [self.output_size])


class DynamicsPredictor(nn.Module):
    def __init__(self, 
                 model_config, 
                 material_config, 
                 dataset_config,
                 device):

        super(DynamicsPredictor, self).__init__()

        self.model_config = model_config
        self.material_config = material_config
        self.dataset_config = dataset_config
        self.device = device
        
        self.n_his = dataset_config['n_his']

        self.nf_particle = model_config['nf_particle']
        self.nf_relation = model_config['nf_relation']
        self.nf_effect = model_config['nf_effect']
        self.nf_physics = model_config['nf_physics']

        self.eps = 1e-6
        self.motion_clamp = 100 

        self.num_materials = len(material_config['material_index'])
        assert self.num_materials == 1, "Only support single material."
        material_params = material_config[dataset_config['materials'][0]]['physics_params']

        material_dim = 0
        for param in material_params:
            if param['use']:
                material_dim += 1

        input_dim = self.n_his * model_config['state_dim'] + \
                    self.n_his * model_config['offset_dim'] + \
                    model_config['attr_dim'] + \
                    model_config['action_dim'] + \
                    model_config['density_dim'] + \
                    material_dim

        self.particle_encoder = Encoder(input_dim, self.nf_particle, self.nf_effect)

        # RelationEncoder
        if model_config['rel_particle_dim'] == -1:
            model_config['rel_particle_dim'] = input_dim

        rel_input_dim = model_config['rel_particle_dim'] * 2 + \
                        model_config['rel_attr_dim'] * 2 + \
                        model_config['rel_group_dim'] + \
                        model_config['rel_distance_dim'] * self.n_his + \
                        model_config['rel_density_dim']
        self.relation_encoder = Encoder(rel_input_dim, self.nf_relation, self.nf_effect)

        # ParticlePropagator
        self.particle_propagator = Propagator(self.nf_effect * 2, self.nf_effect)

        # RelationPropagator
        self.relation_propagator = Propagator(self.nf_effect * 3, self.nf_effect)

        self.non_rigid_predictor = ParticlePredictor(self.nf_effect, self.nf_effect, 3)
        
        if model_config['verbose']:
            print("DynamicsPredictor initialized")
            print("particle input dim: {}, relation input dim: {}".format(input_dim, rel_input_dim))

    # @profile
    def forward(self, state, attrs, Rr, Rs, p_instance, 
            action=None, particle_den=None, obj_mask=None, **kwargs): 

        n_his = self.n_his

        B, N = attrs.size(0), attrs.size(1)  # batch size, total particle num
        n_instance = p_instance.size(2)  # number of instances
        n_p = p_instance.size(1)  # number of object particles (that need prediction)
        n_s = attrs.size(1) - n_p  # number of shape particles that do not need prediction
        n_rel = Rr.size(1)  # number of relations
        state_dim = state.size(3)  # state dimension

        # attrs: B x N x attr_dim
        # state: B x n_his x N x state_dim
        # Rr, Rs: B x n_rel x N
        # memory: B x mem_nlayer x N x nf_memory
        # p_rigid: B x n_instance (deprecated)
        # p_instance: B x n_particle x n_instance
        # physics_param: B x n_particle
        # obj_mask: B x n_particle

        # Rr_t, Rs_t: B x N x n_rel
        Rr_t = Rr.transpose(1, 2).contiguous()
        Rs_t = Rs.transpose(1, 2).contiguous()

        # state_res: B x (n_his - 1) x N x state_dim, state_cur: B x 1 x N x state_dim     
        state_res = state[:, 1:] - state[:, :-1]
        state_cur = state[:, -1:]

        state_res_norm = state_res
        state_cur_norm = state_cur

        # state_norm: B x n_his x N x state_dim
        # [0, n_his - 1): state_residual
        # [n_his - 1, n_his): the current position
        state_norm = torch.cat([state_res_norm, state_cur_norm], 1)
        state_norm_t = state_norm.transpose(1, 2).contiguous().view(B, N, n_his * state_dim)

        # p_inputs: B x N x attr_dim
        p_inputs = attrs

        if self.model_config['state_dim'] > 0:
            # add state to attr
            # p_inputs: B x N x (attr_dim + n_his * state_dim)
            p_inputs = torch.cat([attrs, state_norm_t], 2)

        # instance_center: B x n_instance x (n_his * state_dim)
        instance_center = p_instance.transpose(1, 2).bmm(state_norm_t[:, :n_p])
        instance_center /= torch.sum(p_instance, 1).unsqueeze(-1) + self.eps

        # other inputs
        if self.model_config['offset_dim'] > 0:
            raise NotImplementedError

        # physics
        print(f"kwargs keys: {kwargs.keys()}")
        physics_keys = [k for k in kwargs.keys() if k.endswith('_physics_param')]
        assert len(physics_keys) == 1
        physics_param_og = kwargs[physics_keys[0]]  # (B, phys_dim[i])
        physics_param_og = physics_param_og[:, None, :].repeat(1, n_p, 1)  # (B, N, phys_dim)
        print(f"original physics_param size: ", physics_param_og.size())
        # if physics_param is already a matrix
        physics_param = kwargs[physics_keys[0]].reshape(1, kwargs[physics_keys[0]].shape[-1], 1)
        physics_param_s = torch.zeros(B, n_s, physics_param.shape[2]).to(self.device)
        physics_param = torch.cat([physics_param, physics_param_s], 1)
        print(f"after cat physics_param size: ", physics_param.size())
        p_inputs = torch.cat([p_inputs, physics_param], 2)

        print(f"physics keys: {physics_keys}")
        print(f"B: {B}, N: {n_s}, phys_dim: {physics_param.shape}")
        print(f"physics_param size: ", physics_param.size())
        print(f"physics_param_s size: ", physics_param_s.size())
        print(f"physics_param: {physics_param}, p_inputs: {p_inputs.size()}")

        # action
        if self.model_config['action_dim'] > 0:
            assert action is not None
            p_inputs = torch.cat([p_inputs, action], 2)

        if self.model_config['density_dim'] > 0:
            assert particle_den is not None
            # particle_den: B x N x 1
            particle_den = particle_den[:, None, None].repeat(1, n_p, 1)
            particle_den_s = torch.zeros(B, n_s, 1).to(self.device)
            particle_den = torch.cat([particle_den, particle_den_s], 1)

            # p_inputs: B x N x (... + density_dim)
            p_inputs = torch.cat([p_inputs, particle_den], 2)
        # Finished preparing p_inputs

        # Preparing rel_inputs
        rel_inputs = torch.empty((B, n_rel, 0), dtype=torch.float32).to(self.device)
        if self.model_config['rel_particle_dim'] > 0:
            assert self.model_config['rel_particle_dim'] == p_inputs.size(2)
            # p_inputs_r: B x n_rel x -1
            # p_inputs_s: B x n_rel x -1
            p_inputs_r = Rr.bmm(p_inputs)
            p_inputs_s = Rs.bmm(p_inputs)

            # rel_inputs: B x n_rel x (2 x rel_particle_dim)
            rel_inputs = torch.cat([rel_inputs, p_inputs_r, p_inputs_s], 2)

        if self.model_config['rel_attr_dim'] > 0:
            assert self.model_config['rel_attr_dim'] == attrs.size(2)
            # attr_r: B x n_rel x attr_dim
            # attr_s: B x n_rel x attr_dim
            attrs_r = Rr.bmm(attrs)
            attrs_s = Rs.bmm(attrs)

            # rel_inputs: B x n_rel x (... + 2 x rel_attr_dim)
            rel_inputs = torch.cat([rel_inputs, attrs_r, attrs_s], 2)

        if self.model_config['rel_group_dim'] > 0:
            assert self.model_config['rel_group_dim'] == 1
            # receiver_group, sender_group
            # group_r: B x n_rel x -1
            # group_s: B x n_rel x -1
            g = torch.cat([p_instance, torch.zeros(B, n_s, n_instance).to(self.device)], 1)
            group_r = Rr.bmm(g)
            group_s = Rs.bmm(g)
            group_diff = torch.sum(torch.abs(group_r - group_s), 2, keepdim=True)

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, group_diff], 2)
        
        if self.model_config['rel_distance_dim'] > 0:
            assert self.model_config['rel_distance_dim'] == 3
            # receiver_pos, sender_pos
            # pos_r: B x n_rel x -1
            # pos_s: B x n_rel x -1
            pos_r = Rr.bmm(state_norm_t)
            pos_s = Rs.bmm(state_norm_t)
            pos_diff = pos_r - pos_s

            # rel_inputs: B x n_rel x (... + 3)
            rel_inputs = torch.cat([rel_inputs, pos_diff], 2)

        if self.model_config['rel_density_dim'] > 0:
            assert self.model_config['rel_density_dim'] == 1
            # receiver_density, sender_density
            # dens_r: B x n_rel x -1
            # dens_s: B x n_rel x -1
            dens_r = Rr.bmm(particle_den)
            dens_s = Rs.bmm(particle_den)
            dens_diff = dens_r - dens_s

            # rel_inputs: B x n_rel x (... + 1)
            rel_inputs = torch.cat([rel_inputs, dens_diff], 2)

        # particle encode
        particle_encode = self.particle_encoder(p_inputs)
        particle_effect = particle_encode
        if self.model_config['verbose']:
            print("particle encode:", particle_encode.size())

        # calculate relation encoding
        relation_encode = self.relation_encoder(rel_inputs)
        if self.model_config['verbose']:
            print("relation encode:", relation_encode.size())

        for i in range(self.model_config['pstep']):
            if self.model_config['verbose']:
                print("pstep", i)

            # effect_r, effect_s: B x n_rel x nf
            effect_r = Rr.bmm(particle_effect)
            effect_s = Rs.bmm(particle_effect)

            # calculate relation effect
            # effect_rel: B x n_rel x nf
            effect_rel = self.relation_propagator(
                torch.cat([relation_encode, effect_r, effect_s], 2))
            if self.model_config['verbose']:
                print("relation effect:", effect_rel.size())

            # calculate particle effect by aggregating relation effect
            # effect_rel_agg: B x N x nf
            effect_rel_agg = Rr_t.bmm(effect_rel)

            # calculate particle effect
            # particle_effect: B x N x nf
            particle_effect = self.particle_propagator(
                torch.cat([particle_encode, effect_rel_agg], 2),
                res=particle_effect)
            if self.model_config['verbose']:
                 print("particle effect:", particle_effect.size())

        # non_rigid_motion: B x n_p x state_dim
        non_rigid_motion = self.non_rigid_predictor(particle_effect[:, :n_p].contiguous())
        pred_motion = non_rigid_motion

        pred_pos = state[:, -1, :n_p] + torch.clamp(pred_motion, max=self.motion_clamp, min=-self.motion_clamp)
        if self.model_config['verbose']:
            print('pred_pos', pred_pos.size())

        return pred_pos, pred_motion
