
class yz_ClothRigid: public Scene
{
public:

	yz_ClothRigid(const char* name) : Scene(name)
	{
	}


	// virtual void Initialize(py::array_t<float> scene_params, int thread_idx = 0)
	void Initialize(py::array_t<float> scene_params, 
                    py::array_t<float> vertices,
                    py::array_t<int> stretch_edges,
                    py::array_t<int> bend_edges,
                    py::array_t<int> shear_edges,
                    py::array_t<int> faces,
                    int thread_idx = 0)
	{
		auto ptr = (float *) scene_params.request().ptr;
		int dimx_cloth = ptr[0];
		int dimy_cloth = ptr[1];
		float height_cloth = ptr[2];
		float px_cloth = ptr[3];
		float py_cloth = ptr[4];
		float pz_cloth = ptr[5];

		int dimx_rigid = ptr[6];
		int dimy_rigid = ptr[7];
		int dimz_rigid = ptr[8];
		int numx_rigid = ptr[9];
		int numy_rigid = ptr[10];
		int numz_rigid = ptr[11];

		int num_banana = ptr[12];
		int draw_points = ptr[13];

		/*
	    float x = ptr[0];
	    float y = ptr[1];
	    float z = ptr[2];
	    float dim_x = ptr[3];
	    float dim_y = ptr[4];
	    float dim_z = ptr[5];
	    */

		int sx = dimx_rigid;
		int sy = dimy_rigid;
		int sz = dimz_rigid;

		Vec3 lower(0.0f, height_cloth + g_params.radius, 0.0f);

		int dimx = numx_rigid;
		int dimy = numy_rigid;
		int dimz = numz_rigid;

		float radius = g_params.radius;
		int group = 0;

		if (1)
		{
			char box_path[100];
			strcpy(box_path, getenv("PYFLEXROOT"));
			strcat(box_path, "/data/box.ply");
			// create a basic grid
			for (int y=0; y < dimy; ++y)
				for (int z=0; z < dimz; ++z)
					for (int x=0; x < dimx; ++x)
						CreateParticleShape(
						GetFilePathByPlatform(box_path).c_str(),
						(g_params.radius*0.905f)*Vec3(float(x*sx), float(y*sy), float(z*sz)) + (g_params.radius*0.1f)*Vec3(float(x),float(y),float(z)) + lower,
						g_params.radius*0.9f*Vec3(float(sx), float(sy), float(sz)), 0.0f, g_params.radius*0.9f, Vec3(0.0f), 1.0f, true, 1.0f, NvFlexMakePhase(group++, 0), true, 0.001f);

			// AddPlinth();
		}

		if (1)
		{
			int dimx = dimx_cloth;
			int dimy = dimy_cloth;

			float stretchStiffness = 1.0f;
			float bendStiffness = 0.5f;
			float shearStiffness = 0.7f;

			int clothStart = g_buffers->positions.size();

			// void CreateSpringGrid(Vec3 lower, int dx, int dy, int dz, float radius, int phase, float stretchStiffness, float bendStiffness, float shearStiffness, Vec3 velocity, float invMass)
			CreateSpringGrid(Vec3(px_cloth, py_cloth, pz_cloth), dimx, dimy, 1, radius, NvFlexMakePhase(group++, 0), stretchStiffness, bendStiffness, shearStiffness, Vec3(0.0f), 1.0f);
		
			corner0 = clothStart + 0;
			corner1 = clothStart + dimx-1;
			corner2 = clothStart + dimx*(dimy-1);
			corner3 = clothStart + dimx*dimy-1;

			g_buffers->positions[corner0].w = 0.0f;
			g_buffers->positions[corner1].w = 0.0f;
			g_buffers->positions[corner2].w = 0.0f;
			g_buffers->positions[corner3].w = 0.0f;

			// add tethers
			for (int i=clothStart; i < int(g_buffers->positions.size()); ++i)
			{
				float x = g_buffers->positions[i].x;
				g_buffers->positions[i].y = height_cloth - sinf(DegToRad(0.0f))*x;
				g_buffers->positions[i].x = cosf(DegToRad(0.0f))*x;

				if (i != corner0 && i != corner1 && i != corner2 && i != corner3)
				{
					float stiffness = -0.5f;
					float give = 0.05f;

					CreateSpring(corner0, i, stiffness, give);			
					CreateSpring(corner1, i, stiffness, give);
					CreateSpring(corner2, i, stiffness, give);			
					CreateSpring(corner3, i, stiffness, give);
				}
			}

			g_buffers->positions[corner1] = g_buffers->positions[corner0] + (g_buffers->positions[corner1]-g_buffers->positions[corner0])*0.9f;
			g_buffers->positions[corner2] = g_buffers->positions[corner0] + (g_buffers->positions[corner2]-g_buffers->positions[corner0])*0.9f;
			g_buffers->positions[corner3] = g_buffers->positions[corner0] + (g_buffers->positions[corner3]-g_buffers->positions[corner0])*0.9f;
		}


		char banana_path[100];
		strcpy(banana_path, getenv("PYFLEXROOT"));
		strcat(banana_path, "/data/banana.obj");
		for (int i=0; i < num_banana; ++i)
			CreateParticleShape(GetFilePathByPlatform(banana_path).c_str(), Vec3(0.4f, 8.5f + i*0.25f, 0.25f) + RandomUnitVector()*radius*0.25f, Vec3(1), 0.0f, radius, Vec3(0.0f), 1.0f, true, 0.5f, NvFlexMakePhase(group++, 0), true, radius*0.1f, 0.0f, 0.0f, 1.25f*Vec4(0.875f, 0.782f, 0.051f, 1.0f));
		

		//g_numExtraParticles = 32*1024;		
		g_numSubsteps = 2;
		g_params.numIterations = 4;

		g_params.radius *= 1.0f;
		g_params.staticFriction = 0.7f;
		g_params.dynamicFriction = 0.75f;
		g_params.dissipation = 0.01f;
		g_params.particleCollisionMargin = g_params.radius*0.05f;
		g_params.sleepThreshold = g_params.radius*0.25f;	
		g_params.damping = 0.25f;
		g_params.maxAcceleration = 400.0f;

		g_windStrength = 0.0f;

		// draw options
		g_drawPoints = draw_points;
		g_drawMesh = 1 - draw_points;

		g_emitters[0].mEnabled = true;
		g_emitters[0].mSpeed = (g_params.radius*2.0f/g_dt);
	}

	virtual void Update(py::array_t<float> update_params)
	{
        // update action
        auto ptr = (float *) update_params.request().ptr;

        float dx = ptr[0];
	    float dy = ptr[1];
	    float dz = ptr[2];

	    g_buffers->positions[corner0].x += dx;
	    g_buffers->positions[corner0].y += dy;
	    g_buffers->positions[corner0].z += dz;

	    g_buffers->velocities[corner0].x = dx / g_dt;
        g_buffers->velocities[corner0].y = dy / g_dt;
        g_buffers->velocities[corner0].z = dz / g_dt;

        g_buffers->positions[corner1].x += dx;
	    g_buffers->positions[corner1].y += dy;
	    g_buffers->positions[corner1].z += dz;

	    g_buffers->velocities[corner1].x = dx / g_dt;
        g_buffers->velocities[corner1].y = dy / g_dt;
        g_buffers->velocities[corner1].z = dz / g_dt;

        g_buffers->positions[corner2].x += dx;
	    g_buffers->positions[corner2].y += dy;
	    g_buffers->positions[corner2].z += dz;

	    g_buffers->velocities[corner2].x = dx / g_dt;
        g_buffers->velocities[corner2].y = dy / g_dt;
        g_buffers->velocities[corner2].z = dz / g_dt;

        g_buffers->positions[corner3].x += dx;
	    g_buffers->positions[corner3].y += dy;
	    g_buffers->positions[corner3].z += dz;

	    g_buffers->velocities[corner3].x = dx / g_dt;
        g_buffers->velocities[corner3].y = dy / g_dt;
        g_buffers->velocities[corner3].z = dz / g_dt;
	}

	int corner0, corner1, corner2, corner3;
};
