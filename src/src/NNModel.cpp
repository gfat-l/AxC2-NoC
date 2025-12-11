/*
 * NN-Noxim - the NoC-based ANN Simulator
 *
 * (C) 2018 by National Sun Yat-sen University in Taiwan
 *
 * This file contains the implementation of loading NN model
 */

#include "NNModel.h"
#include <iomanip>
#include <math.h>
#include <ctime>
#include <string>

NNModel::NNModel()
{
	for(int i=0;i<NoximGlobalParams::tdm;i++){
		all_leyer_type.push_back(deque< char >{});
		all_leyer_size.push_back(deque< deque< int > >{});
		mapping_table.push_back(deque< int >{});
		each_layer_num.push_back(deque< int >{});
		//all_conv_weight.push_back(deque <deque<deque< deque< float >>>>{});
		all_conv_weight.push_back(deque <deque<deque< deque< long long int >>>>{});
		all_leyer_ID_Group.push_back(deque< deque< int > >{});
		//all_data_in.push_back(deque< deque< float > >{});
		all_data_in.push_back(deque< deque< long long int > >{});
		Group_table.push_back(deque < deque< NeuInformation > >{});
		//all_conv_bias.push_back(deque<deque <float>>{});
		all_conv_bias.push_back(deque<deque <long long int>>{});
		all_conv_coord.push_back(deque <deque<deque<int>>>{});
		all_pool_coord.push_back(deque <deque <deque<int>>>{});
		//all_bn_weight.push_back(deque<deque<deque<float>>>{});
		all_bn_weight.push_back(deque<deque<deque<long long int>>>{});
		layer_scale.push_back(deque<int>{});

		//''''''MODIFY BY LCZ'''''''
		all_layer_approx_level_table.push_back(deque< deque< int > >{});
		all_layer_approx.push_back(deque< deque< int > >{});
		drop_rate_new.push_back(deque< int >{});

	}
}

bool NNModel::load()//M_fname Useless tytyty
{
	cout<<"model file loading (filename: " << NoximGlobalParams::NNmodel_filename <<" and "<<NoximGlobalParams::NNmodel_filename1<< ")..."<< endl;		//** 2018.09.02 edit by Yueh-Chi,Yang **//
	string NNmodel_filename_tmp[3] = {NoximGlobalParams::NNmodel_filename,NoximGlobalParams::NNmodel_filename1,NoximGlobalParams::NNmodel_filename2};
	string NNweight_filename_tmp[3] = {NoximGlobalParams::NNweight_filename,NoximGlobalParams::NNweight_filename1,NoximGlobalParams::NNweight_filename2};
	string approx_filename_tmp[3] = {NoximGlobalParams::NNapprox_filename,NoximGlobalParams::NNapprox_filename1,NoximGlobalParams::NNapprox_filename2};
	string approx_level_table_temp[3] = {NoximGlobalParams::NNapprox_level_tablefilename,NoximGlobalParams::NNapprox_level_tablefilename1,NoximGlobalParams::NNapprox_level_tablefilename2};
	string drop_ratefile= NoximGlobalParams::NNapprox_dropratefile;
	//cout<<"model file loading (filename: " << NoximGlobalParams::NNmodel_filename << ")..."<< endl;		//** 2018.09.02 edit by Yueh-Chi,Yang **//
	//string NNmodel_filename_tmp[1] = {NoximGlobalParams::NNmodel_filename};
	//string NNweight_filename_tmp[1] = {NoximGlobalParams::NNweight_filename};
	for(int wr=0; wr<NoximGlobalParams::tdm;wr++){
		ifstream fin(NNmodel_filename_tmp[wr], ios::in); 
		ifstream fin1(approx_filename_tmp[wr], ios::in);
		ifstream fin2(approx_level_table_temp[wr], ios::in);
		ifstream fin3(drop_ratefile, ios::in);
		
		//all_leyer_type.clear();
		//all_leyer_size.clear();
		char temp_type[20], temp_sv_pad[20], temp_actfun[10];
		int temp;
		int temp_c_x, temp_c_y, temp_z, temp_num, temp_std, temp_x, temp_y, temp_pad, temp_channels, temp_c_z, weight_scale,approx_threshold,bn,output_scale;
		int input_size, output_size;

		//""""MODIFY BY LCZ"""
		int level[10] = {0};
		int app1[4] = {0};
		//END MODIFY 
		
		deque< deque< int > > conv;
		deque< deque< int > > pool;
		int all_Nue=0;
		// *****************all layer Neu_num setting*******************
		cout<<endl;
		cout<<"layer_ID |    type | Neu_num |       X |       Y | channel |   C/P_X |   C/P_Y |    C/P_Z |  stride | padding | weight_scale | act_fun |"<<endl;
		cout<<"--------------------------------------------------------------------------------------------------------------------------------------"<<endl;
		while(fin1 >> temp_type){
			if (!strcmp( temp_type, "Dense") || !strcmp( temp_type, "Convolution") || !strcmp( temp_type, "Pooling"))
			{
				deque< int > temp_leyer_size;
				char line[256];
				fin1.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d %d", app1,app1+1,app1+2,app1+3);
				temp_leyer_size.push_back(app1[3]);
				temp_leyer_size.push_back(app1[2]);
				temp_leyer_size.push_back(app1[1]);
				temp_leyer_size.push_back(app1[0]);
				all_layer_approx[wr].push_back(temp_leyer_size);
				each_layer_num[wr].push_back(0);
			}
			else if (!strcmp( temp_type, "%"))
			{
				char line[256];
				fin1.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout << "load approximate threshold error!" << endl;
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin1.getline(line, sizeof(line) - 1);
			}
		}
		// for(int i = 0;i < 10;++i){
		// 	for(int j = 0;j < 4;++j){
		// 		cout << all_layer_approx[wr][i][j] << " ";
		// 	}
		// 	cout << endl;
		// }

//"""MODIFY BY LCZ"""
		while(fin2 >> temp_type){
			if (!strcmp( temp_type, "Dense") || !strcmp( temp_type, "Convolution") || !strcmp( temp_type, "Pooling"))
			{
				deque< int > temp_leyer_size;
				char line[256];
				fin2.getline(line, sizeof(line) - 1);
				// sscanf(line, "%d", level);
				sscanf(line, "%d %d %d %d %d %d", level,level+1,level+2,level+3,level+4,level+5);
				temp_leyer_size.push_back(level[0]);
				temp_leyer_size.push_back(level[1]);
				temp_leyer_size.push_back(level[2]);
				temp_leyer_size.push_back(level[3]);
				temp_leyer_size.push_back(level[4]);
				temp_leyer_size.push_back(level[5]);
				all_layer_approx_level_table[wr].push_back(temp_leyer_size);
			}
			else if (!strcmp( temp_type, "%"))
			{
				char line[256];
				fin2.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout << "load approximate level error!" << endl;
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin2.getline(line, sizeof(line) - 1);
			}
		}
		while(fin3 >> temp_type){
			drop_rate_new[wr].push_back(stoi(temp_type));
		}
		// for(int i = 0;i < 10;++i){
		// 	for(int j = 0;j < 6;++j){
		// 		cout << all_layer_approx_level_table[wr][i][j]<< " ";
		// 	}
		// 	cout << endl;
		// }
//END MODIFY

		while(fin >> temp_type){


			if (!strcmp( temp_type, "Dense"))
			{
				all_leyer_type[wr].push_back('f');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %s %d %d %d", &temp, temp_actfun, &weight_scale, &approx_threshold, &output_scale);
				
				deque< int > temp_leyer_size;
				temp_leyer_size.push_back(temp);
				if(!strcmp( temp_actfun, "relu"))
					temp_leyer_size.push_back(RELU);
				else if(!strcmp( temp_actfun, "tanh"))
					temp_leyer_size.push_back(TANH);
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_leyer_size.push_back(SIGMOID);
				else if(!strcmp( temp_actfun, "softmax"))
					temp_leyer_size.push_back(SOFTMAX);
				else
					temp_leyer_size.push_back(10);
				temp_leyer_size.push_back(weight_scale);
				temp_leyer_size.push_back(approx_threshold);
				temp_leyer_size.push_back(output_scale);

				all_leyer_size[wr].push_back(temp_leyer_size);
				all_Nue+=temp;
				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<    setw(8)<<"Fully"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
					<<                   setw(10)<<" |"<<            setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<            setw(10)<<" |"<<            setw(10)<<" |"
					<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<setw(10)<<temp_leyer_size[2]<<" |"<<setw(10)<<temp_actfun<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Input"))
			{
				all_leyer_type[wr].push_back('i');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d %d", &temp_x, &temp_y, &temp_z, &weight_scale);
				temp = temp_x * temp_y*temp_z;
				deque< int > temp_leyer_size;
				temp_leyer_size.push_back(temp);
				temp_leyer_size.push_back(temp_x);
				temp_leyer_size.push_back(temp_y);
				temp_leyer_size.push_back(temp_z);
				temp_leyer_size.push_back(weight_scale);

				all_leyer_size[wr].push_back(temp_leyer_size);
				//all_Nue+=temp;
				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<           setw(8)<<"Input"<<" |"<<setw(10)<<" |"
					<<setw(8)<<temp_leyer_size[1]<<" |"<<setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<<                   setw(10)<<" |"
					<<                   setw(10)<<" |"<<                   setw(10)<<" |"<< setw(10)<<temp_leyer_size[4]<<" |"<<setw(10)<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "Convolution"))
			{
				all_leyer_type[wr].push_back('c');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d %d %d %d %d %d %s %d %d %d %d",&temp_x,&temp_y, &temp_channels, &temp_c_x, &temp_c_y,&temp_c_z, &temp_std, &temp_pad, temp_actfun, &weight_scale,&approx_threshold, &bn, &output_scale);
				deque< int > temp_leyer_size;
				temp = temp_x *temp_y*temp_channels;
				temp_leyer_size.push_back(temp);  //The size of the convolution layer
				temp_leyer_size.push_back(temp_x);
				temp_leyer_size.push_back(temp_y);
				temp_leyer_size.push_back(temp_channels);
				temp_leyer_size.push_back(temp_c_x);
				temp_leyer_size.push_back(temp_c_y);
				temp_leyer_size.push_back(temp_c_z);
				temp_leyer_size.push_back(temp_std);
				temp_leyer_size.push_back(temp_pad);
				temp_leyer_size.push_back(weight_scale);
				temp_leyer_size.push_back(approx_threshold);
				temp_leyer_size.push_back(bn);
				temp_leyer_size.push_back(output_scale);
				if(!strcmp( temp_actfun, "relu"))
					temp_leyer_size.push_back(RELU);
				else if(!strcmp( temp_actfun, "tanh"))
					temp_leyer_size.push_back(TANH);
				else if(!strcmp( temp_actfun, "sigmoid"))
					temp_leyer_size.push_back(SIGMOID);
				else if(!strcmp( temp_actfun, "softmax"))
					temp_leyer_size.push_back(SOFTMAX);
				

				all_leyer_size[wr].push_back(temp_leyer_size);
				all_Nue+=temp;
				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<     setw(8)<<"Convol"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
				<<      setw(8)<<temp_leyer_size[1]<<" |"<<   setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
				<<      setw(8)<<temp_leyer_size[4]<<" |"<<   setw(8)<<temp_leyer_size[5]<<" |"<<setw(8)<<temp_leyer_size[6]<<" |"	
				<<      setw(8)<<temp_leyer_size[7]<<" |"<<   setw(8)<<temp_leyer_size[8]<<" |"<<setw(10)<<temp_leyer_size[9]<<" |"<<setw(10)<<temp_actfun<<" |"<<endl;
			
			}else if (!strcmp( temp_type, "Pooling"))
			{
				all_leyer_type[wr].push_back('p');
				char line[256];
				fin.getline(line, sizeof(line) - 1);
				sscanf(line, "%d %d %d %d %d %d %s %d", &temp_x,&temp_y, &temp_channels, &temp_c_x, &temp_c_y, &temp_std, temp_actfun,&approx_threshold);
				deque< int > temp_leyer_size;
				temp = temp_x *temp_y*temp_channels;
				temp_leyer_size.push_back(temp);//The size of the pooling layer
				temp_leyer_size.push_back(temp_x);
				temp_leyer_size.push_back(temp_y);
				temp_leyer_size.push_back(temp_channels);
				temp_leyer_size.push_back(temp_c_x);
				temp_leyer_size.push_back(temp_c_y);
				temp_leyer_size.push_back(temp_std);
				temp_leyer_size.push_back(approx_threshold);
				
				if(!strcmp( temp_actfun, "average"))
					temp_leyer_size.push_back(AVERAGE);
				else if(!strcmp( temp_actfun, "maximum"))
					temp_leyer_size.push_back(MAXIMUM);

				all_leyer_size[wr].push_back(temp_leyer_size);
				all_Nue+=temp;
				cout<<setw(8)<<all_leyer_type[wr].size()-1<<" |"<<     setw(8)<<"Pooling"<<" |"<<setw(8)<<temp_leyer_size[0]<<" |"
				<<      setw(8)<<temp_leyer_size[1]<<" |"<<   setw(8)<<temp_leyer_size[2]<<" |"<<setw(8)<<temp_leyer_size[3]<<" |"
				<<      setw(8)<<temp_leyer_size[4]<<" |"<<   setw(8)<<temp_leyer_size[5]<<" |"<<            setw(10)<<" |"<<setw(8)<<temp_leyer_size[6]<<" |"	
				<<                   setw(10)<<" |"<<         setw(12)<<" |"<<         setw(10)<<temp_actfun<<" |"<<endl;
			}
			else if (!strcmp( temp_type, "%"))
			{
			
				char line[256];
				fin.getline(line, sizeof(line) - 1);
			}
			else
			{
				cout<<"!!Error model format: "<<temp_type<<" !!"<<endl;
				char line[256];
				fin.getline(line, sizeof(line) - 1);
			}

		}
		//for( int i=0; i<5;i++){
		//  cout<<all_leyer_type[i]<<"-------";
		//}

		cout<<"model all_leyer complete"<<endl;
		cout<<"all neu:"<<all_Nue<<endl;
		cout <<  "max_ID_LAYER"<<all_leyer_size[0].size() << endl; 
		fin.close();
		input_size=all_leyer_size[wr].front()[0];
		output_size=all_leyer_size[wr].back()[0];
		for(int xx=0; xx<all_leyer_size[wr].size(); xx++){
			layer_scale[wr].push_back(0);
		}
		cout<<input_size<<"|"<<output_size<<endl;
	

		//******************mapping information prepare************************
		mapping_table[wr].clear();
		mapping_table[wr].assign( NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y, -1 );
    
		cout<<"ALgorithm: "<< NoximGlobalParams::mapping_algorithm<<endl;
		if(!strcmp(NoximGlobalParams::mapping_algorithm, "random") )
		{
			srand( time(NULL) );
			for( int i = 0; i < mapping_table[wr].size() ; i++ )
			{
				while(1)
				{
					int map_point = rand() % mapping_table[wr].size();
					if( mapping_table[wr][map_point]==-1 )
					{
						mapping_table[wr][map_point] = i;
						break;
					}

				}
			}
		}
		else if(!strcmp(NoximGlobalParams::mapping_algorithm, "dir_x") ){
			for( int i = 0; i < mapping_table[wr].size() ; i++ ){	//dir_x mapping
				if(i%2==0)
				mapping_table[wr][i]=i*3;//mapping_table[wr][i]=i;
				else 
				mapping_table[wr][i]=64-i*3;
			}
		}
		else if(!strcmp(NoximGlobalParams::mapping_algorithm, "dir_y") )	
			for( int i = 0; i < mapping_table[wr].size() ; i++ )	//dir_y mapping
				mapping_table[wr][i]= (i%NoximGlobalParams::mesh_dim_x)*NoximGlobalParams::mesh_dim_x + (i/NoximGlobalParams::mesh_dim_x);	
		else if(!strcmp(NoximGlobalParams::mapping_algorithm, "table") )	
		{
			ifstream fin_m(NoximGlobalParams::mapping_table_filename, ios::in);						//** 2018.09.02 edit by Yueh-Chi,Yang **
			cout<<"mapping file loading (filename: " << NoximGlobalParams::mapping_table_filename << ")..."<< endl;		//** 2018.09.02 edit by Yueh-Chi,Yang **
			while(!fin_m.eof()){
				char line[256];
				fin_m.getline(line, sizeof(line) - 1);
				if (line[0] != '\0') {
						if (line[0] != '%') {
						int ID_Group, ID_PE;
						sscanf(line, "%d %d", &ID_Group, &ID_PE);
						mapping_table[wr][ID_Group] = ID_PE;
						}
				}
			}
		}

		else
		{
			ifstream fin_m(NoximGlobalParams::mapping_table_filename, ios::in);                                             //** 2018.09.02 edit by Yueh-Chi,Yang **
					cout<<"mapping file loading (filename: " << NoximGlobalParams::mapping_table_filename << ")..."<< endl;         //** 2018.09.02 edit by Yueh-Chi,Yang **
					while(!fin_m.eof()){
							char line[256];
							fin_m.getline(line, sizeof(line) - 1);
							if (line[0] != '\0') {
									if (line[0] != '%') {
											int ID_Group, ID_PE;
											sscanf(line, "%d %d", &ID_Group, &ID_PE);
											mapping_table[wr][ID_Group] = ID_PE;
									}
							}
					}

			//cout<<"Error mapping algorithm!!"<<endl;
			//exit(1);
		}
		/*------Debugging--------*/
		//cout<<"Mapping Table"<<endl;
		//for( int i =0; i< mapping_table[wr].size();i++){
		//	cout<< mapping_table[wr].at(i)<<"--";
		//}
		//cout<<endl;
		/*-----------------------*/
		cout<<"Model mapping table: "<<mapping_table[wr].size()<<endl;
		cout<<"maping complete"<<endl;

		// ******************temp_Group_table setting**********************
		int temp_ID_Neu=0;
		int temp_layer=1; //Layer id starts from layer 1 which is convolution layer
		int temp_ID_In_layer=0;
		int temp_ID_Group=0;
		int temp_ID_In_Group=0;
		//float temp_w;
		long long int temp_w;
		int temp_ID_conv = 0;
		int temp_ID_pool =-1;

		//deque <deque<float>> temp_conv_weight;
		//deque <deque<float>> temp_bn_w;
		//deque<deque<deque<float>>> temp_conv_weight_layer;
		deque <deque<long long int>> temp_conv_weight;
		deque <deque<long long int>> temp_bn_w;
		deque<deque<deque<long long int>>> temp_conv_weight_layer;
		int temp_layer_maxID = all_leyer_size[wr][temp_layer][0]; //Starting with layer 1
		//cout<<"Max id: "<<temp_layer_maxID<<endl;
		Group_table[wr].clear();
		deque < NeuInformation > temp_Group_table;
		temp_Group_table.clear();
		deque < int > temp_leyer_ID_Group;
		temp_leyer_ID_Group.clear();
		
		//int prevLayer_size;
		int kernel_size;
		//deque <float> temp_bias;
		deque <long long int> temp_bias;
		ifstream fin_w(NNweight_filename_tmp[wr], ios::in);	
		cout<<"weight file loading (filename: " << NNweight_filename_tmp[wr]<< ")..."<<endl;	//** 2018.09.02 edit by Yueh-Chi,Yang **//
		
		//Save convolution weights and bias for each filter
		//deque<float> temp_weights;
		//deque<float> temp_bias_conv;
		//deque<float> temp_bn_weight;
		deque<long long int> temp_weights;
		deque<long long int> temp_bias_conv;
		deque<long long int> temp_bn_weight;

		for(int i =0; i< all_leyer_type[wr].size(); i++)
		{
		if(all_leyer_type[wr][i] == 'c')
		{
			kernel_size = all_leyer_size[wr][i][4] * all_leyer_size[wr][i][5];
			temp_conv_weight.clear();
			temp_bias_conv.clear();
			temp_conv_weight_layer.clear();
			temp_bn_weight.clear();
			temp_bn_w.clear();
			for(int j=0; j< all_leyer_size[wr][i][3];j++)
			{	
				for(int q=0;q<all_leyer_size[wr][i][6];q++)
				{
					temp_weights.clear();
					for(int l=0; l<kernel_size;l++)
					{
						fin_w >> temp_w;
						temp_weights.push_back(temp_w);
					}
					temp_conv_weight.push_back(temp_weights);//Convolution kernel weights
				}
				temp_conv_weight_layer.push_back(temp_conv_weight);
				temp_conv_weight.clear();
			}
				all_conv_weight[wr].push_back(temp_conv_weight_layer);
				temp_conv_weight_layer.clear();
			for( int m =0; m < all_leyer_size[wr][i][3];m++)
			{
				fin_w >> temp_w;
				temp_bias_conv.push_back(temp_w);   //Bias for each filter   
			}
			all_conv_bias[wr].push_back(temp_bias_conv);
			// 加上bn层的权重
			if(all_leyer_size[wr][i][11]){
				for(int qs=0; qs<4; qs++){
					for(int j=0; j< all_leyer_size[wr][i][3];j++)
					{
						fin_w >> temp_w;
						temp_bn_weight.push_back(temp_w);		
					}
					temp_bn_w.push_back(temp_bn_weight);
					temp_bn_weight.clear();
				}
				all_bn_weight[wr].push_back(temp_bn_w);
				temp_bn_w.clear();
			}
		}

		}

		/*--------------Debugging---------------------------*/
		/*cout<< all_conv_bias[wr].size()<<"--"<<all_conv_bias[wr][1].size()<<endl;
		for( int p=0; p< all_conv_bias[wr][1].size();p++)
		{
			cout<<all_conv_bias[wr][1][p]<<"-----";
		}*/
		/*cout<<endl;
		cout<<all_conv_weight[wr].size()<<endl;
		
		cout<<all_conv_weight[wr][0].size()<<endl;
		cout<<all_conv_weight[wr][1].size()<<endl;
		cout<<all_conv_weight[wr][0][0].size()<<endl;
		cout<<all_conv_weight[wr][1][0].size()<<endl;
		cout<<all_conv_weight[wr][0][0][0].size()<<endl;
		cout<<all_conv_weight[wr][1][0][0].size()<<endl;
		for( int p=0; p< all_conv_weight[wr][1][0].size();p++)
		{
			for(int q=0; q<all_conv_weight[wr][1][0][p].size();q++)
			{
				cout<<all_conv_weight[wr][1][0][p][q]<<"---";
			}
			
		}
		cout<<endl;*/
		/*--------------------------------------------------*/
		
		
		while(1)
		{
			NeuInformation NeuInfo;

			NeuInfo.ID_Neu = temp_ID_Neu;

			if(temp_ID_Neu < temp_layer_maxID)
			{									
				NeuInfo.ID_layer = temp_layer;
				NeuInfo.ID_In_layer = temp_ID_In_layer;
				if(all_leyer_type[wr][temp_layer] == 'c')
				{	
					NeuInfo.ID_conv = temp_ID_conv;
					//cout<<"("<< temp_ID_Neu<<"--"<<NeuInfo.ID_conv<<")--";

				}else if (all_leyer_type[wr][temp_layer] == 'p' )
				{
					NeuInfo.ID_pool = temp_ID_pool;
					//cout<<"("<< temp_ID_Neu<<"--"<<NeuInfo.ID_pool<<")--";

				}

				if( temp_ID_In_Group >=  NoximGlobalParams::group_neu_num){					//** 2018.09.01 edit by Yueh-Chi,Yang **// change group
					Group_table[wr].push_back(temp_Group_table);
						temp_Group_table.clear();

					temp_ID_In_Group = 0;
					temp_leyer_ID_Group.push_back(temp_ID_Group);
					temp_ID_Group++;
				}
				NeuInfo.ID_In_Group = temp_ID_In_Group;
				NeuInfo.ID_Group = temp_ID_Group;
				/*----------------Debugging-----------------*/
				/*if( temp_layer == 1 || temp_layer == 3)
				{
					cout<< "ID Conv : "<<temp_ID_conv<<"--"; 
				}else if( temp_layer == 2 || temp_layer == 4)
				{
					cout<<"ID_pool: "<<temp_ID_pool<<"--";
				}*/
				/*------------------------------------------*/
			}
			else{													//change layer and group
				//cout<<"Temp id in layer: "<<temp_ID_In_layer<<endl;
				Group_table[wr].push_back(temp_Group_table);
				temp_Group_table.clear();
						
				NeuInfo.ID_layer = ++temp_layer;
				temp_leyer_ID_Group.push_back(temp_ID_Group);
				all_leyer_ID_Group[wr].push_back(temp_leyer_ID_Group);
				temp_leyer_ID_Group.clear();
				NeuInfo.ID_Group = ++temp_ID_Group;

				if(temp_ID_Neu >= all_Nue) break;

				if(temp_ID_Group >= NoximGlobalParams::mesh_dim_x*NoximGlobalParams::mesh_dim_y)
					cout<<"error group_size or NoC_size" <<endl;						//output error!! 
				
				temp_ID_In_layer = 0;
				temp_ID_In_Group = 0;
				NeuInfo.ID_In_layer = temp_ID_In_layer;
				NeuInfo.ID_In_Group = temp_ID_In_Group;
				temp_layer_maxID += all_leyer_size[wr][temp_layer][0];


				if(all_leyer_type[wr][temp_layer] == 'c' )
				{	
					
					NeuInfo.ID_conv = ++temp_ID_conv;
					//cout<<"("<< temp_ID_Neu<<"--"<<NeuInfo.ID_conv<<")--";

				}else if (all_leyer_type[wr][temp_layer] == 'p' )
				{
					NeuInfo.ID_pool = ++temp_ID_pool;
					//cout<<"("<< temp_ID_Neu<<"--"<<NeuInfo.ID_pool<<")--";

				}
				temp_bias.clear();
				if(all_leyer_type[wr][temp_layer]=='f')
				{
					for(int k=0;k<all_leyer_size[wr][temp_layer][0];k++)
					{
						fin_w >> temp_w;
						temp_bias.push_back(temp_w);
						
					}
					/*-------------Debugging------------*/
					//cout<<temp_bias.back()<<endl;
					//cout<<temp_bias.front()<<endl;
					//cout<<"Size of temp_bias: "<<temp_bias.size()<<"--"<<temp_layer<<endl;
					/*----------------------------------*/
				}
				
			}

			NeuInfo.Type_layer = all_leyer_type[wr][NeuInfo.ID_layer];

			if( NeuInfo.Type_layer == 'f')
			{
				for( int i=0 ; i < ( all_leyer_size[wr][ temp_layer-1 ][0]) ; i++ )	
				{
					fin_w>>temp_w;
					NeuInfo.weight.push_back(temp_w);
				}
				NeuInfo.weight.push_back(temp_bias[NeuInfo.ID_In_layer]); //Include bias
				/*------Debugging------*/
				//cout<<NeuInfo.weight.size()<<"-"<<NeuInfo.weight.front()<<"-"<<NeuInfo.weight.back()<<endl;
				/*---------------------*/
			}
			//cout<<endl;
			//***********mapping****************
			//all_conv_weight[wr].push_back(temp_conv_weight);
			temp_Group_table.push_back(NeuInfo);
			temp_ID_In_Group++;
			temp_ID_In_layer++;
			temp_ID_Neu++;
		}
		/*cout<<endl<<"Convolution weight: "<<endl;
		for( int i=0; i<all_leyer_type.size(); i++)
		{
		if( all_leyer_type[i] == 'c')
		{
			for( int j= 0; j< (all_leyer_size[wr][i][3] * all_leyer_size[wr][i][4] *all_leyer_size[wr][i][5] + all_leyer_size[wr][i][3]); j++)
			{ 
			fin_w >> temp_w;
			temp_conv_weight.push_back(temp_w); //Convolution weight and then bias value
			cout<<temp_w<< "----";
				}		
		}
		all_conv_weight[wr].push_back(temp_conv_weight);
		} */	
		fin_w.close();
		
		//deque<float>().swap(temp_conv_weight);
		deque<NeuInformation>().swap(temp_Group_table);
		deque<int>().swap(temp_leyer_ID_Group);
		cout<<"model & group complete"<<endl;
		
		
		/*-------------------Debugging---------------------*/
		/*
		cout<< Group_table[wr][1][0].Type_layer<<endl;
		cout<< Group_table[wr][2][0].Type_layer<<endl;;
		
		cout<<"Group Table"<<endl;
		cout<<"Neuron ID"<<endl;
		for( int i =0; i< Group_table[wr].size();i++){
			for( int j=0; j< Group_table[wr][i].size();j++ ){
				cout<< Group_table[wr][i][j].ID_Neu<<"--";

			}	
		}
		cout<<endl;
		
		cout<<"Id in Group"<<endl;
		for( int i =0; i< Group_table[wr].size();i++){
			for( int j=0; j< Group_table[wr][i].size();j++ ){
				cout<< Group_table[wr][i][j].ID_In_Group<<"--";

			}	
		}
		cout<<endl;
		cout<<"Id in layer"<<endl;
		for( int i =0; i< Group_table[wr].size();i++){
			for( int j=0; j< Group_table[wr][i].size();j++ ){
				cout<< Group_table[wr][i][j].ID_In_layer<<"--";

			}	
		}
		cout<<endl;
		
		//Reverse Eng
		cout<<"All Layer ID Group"<<endl;
		
		
		for( int i =0; i< all_leyer_ID_Group[wr].size();i++){
			for( int j=0; j< all_leyer_ID_Group[wr][i].size();j++ ){
				cout<< all_leyer_ID_Group[wr][i][j]<<"--";

			}
			
		}
		
		cout<<"All layer id group size: "<<all_leyer_ID_Group[wr].size()<<endl;
		//cout<<"Last layer: "<<all_leyer_ID_Group[wr][6].size()<<endl;
		//cout<<"First layer: "<<all_leyer_ID_Group[wr][0].size()<<endl;
		cout<<"Group table size: "<<Group_table[wr].size()<<endl;
		cout<<"all layer type size: "<<all_leyer_type[wr].size()<<endl;
		cout<<"all layer size: "<<all_leyer_size[wr].size()<<endl;
		
		cout<<endl;
		//Reverse Eng
		cout<<"All layer Size"<<endl;
		for( int i =0; i< all_leyer_size[wr].size();i++){
			for( int j=0; j< all_leyer_size[wr][i].size();j++ ){
				cout<< all_leyer_size[wr][i][j]<<"--";

			}
			
		}
		cout<<endl;

		//Reverse Eng
		cout<<"All layer Type"<<endl;

		for( int i =0; i< all_leyer_type[wr].size();i++)
			{
				cout<< all_leyer_type[wr][i]<<"--";	
			}
		cout<<endl; */
		/*-----------------------------------*/
		//******************print floorplan****************

		cout<<"Hardware floorplan:"<<endl;
		cout<<"  ";	
		for(int i=0;i<NoximGlobalParams::mesh_dim_x;i++)
		cout<<"-----";	
		cout<<"-"<<endl;	
		for(int j=0;j<NoximGlobalParams::mesh_dim_y;j++)
		{
			cout<<"  |";
			for(int i=0;i<NoximGlobalParams::mesh_dim_x;i++)
			{

				int local_id = j*NoximGlobalParams::mesh_dim_x + i;
				int x;
					for(x=0;x<mapping_table[wr].size();x++)
					{
						if(mapping_table[wr][x]==local_id) break;
					}
					if(x<Group_table[wr].size())
					{
						int temp_lay=Group_table[wr][x][0].ID_layer;
						cout<<setw(3)<<temp_lay<<" |";
						each_layer_num[wr][temp_lay-1]++;
					}
					else
						cout<<setw(3)<<" "<<" |";

			}
			cout<<endl<<"  ";
			for(int i=0;i<NoximGlobalParams::mesh_dim_x;i++)
			cout<<"-----";	
			cout<<"-"<<endl;
		}
		//siyue modify
		/*
		for(int i=0; i<each_layer_num[wr].size(); i++){
			cout<<each_layer_num[wr][i]<<endl;
		}
		*/
		//***********input setting***************

		if(wr==0){
			fstream fin_in(NoximGlobalParams::NNinput_filename, ios::in); 
			//float temp_in;
			long long int temp_in;
			int i = -1;
			//deque< float > temp_data_in;
			deque< long long int > temp_data_in;
			temp_data_in.clear();

			while(fin_in >> temp_in){
				i++;
				temp_data_in.push_back(temp_in);
				if(i==input_size-1)
				{
					all_data_in[0].push_back(temp_data_in);
					temp_data_in.clear();
					i=-1;
				}
			}

			//deque<float>().swap(temp_data_in);
			deque<long long int>().swap(temp_data_in);
			fin_in.close();

			cout<<"all_data_in[0].size(): "<<all_data_in[0].size()<<endl;
			cout<<"load input complete"<<endl;
		}
		
		/*--------Debugging-----------------*/
		//cout<<"All data in"<<endl;
		//for( int i =0; i< all_data_in[0].size();i++){
		//	for( int j=0; j< all_data_in[0][i].size();j++ ){
		//       cout<<"( "<<j<<": "<< all_data_in[0][i][j]<<")--";
		//	}	
		//}
		//cout<<endl;
		//cout<<"Size of input: "<<all_data_in[0][0].size()<<endl;
		/*----------------------------------*/
		
		//******************Save the coordinates for 2d Convolution and Pooling****************
		all_conv_coord[wr].clear();
		all_pool_coord[wr].clear();
		deque< int> temp_cell;
		deque < deque<int>> temp_matrix;
		deque < deque < deque< int > > > pad_index_matrix;
		deque < int > pad_row_index;
		deque <deque<int> > pad_col_index;
		for(int ab =0; ab< all_leyer_size[wr].size(); ab++)
		{
			if(all_leyer_type[wr][ab] == 'c')
			{
				int coord_x = all_leyer_size[wr][ab][1];
				int coord_y = all_leyer_size[wr][ab][2];
				int coordPrev_x = all_leyer_size[wr][ab-1][1];
				int coordPrev_y = all_leyer_size[wr][ab-1][2];
				int kernel_x = all_leyer_size[wr][ab][4];
				int kernel_y = all_leyer_size[wr][ab][5];
				int kernel_z = all_leyer_size[wr][ab][6];
				int stride = all_leyer_size[wr][ab][7];
				int padding = all_leyer_size[wr][ab][8];

				if(padding == 0){
					for(int aa =0; aa< coord_x; aa++)
					{
						for(int bb =0; bb < coord_y; bb++)
						{
							for(int cc =0; cc< kernel_z; cc++)
							{
								for(int dd =0; dd< kernel_x; dd++)
								{
									for(int ee =0; ee<kernel_y; ee++)
									{
										temp_cell.push_back((aa*stride+dd)*coordPrev_y + (ee+ bb*stride) + cc*coordPrev_x*coordPrev_y);
									}
								}
							}
							temp_matrix.push_back(temp_cell);
							temp_cell.clear();
						}
					}
					all_conv_coord[wr].push_back(temp_matrix);
					temp_matrix.clear();
				}else if(padding>0)
				{
					for(int zc = 0; zc < kernel_z; zc++){
						for(int za = 0;za < coordPrev_x+2*padding; za++){
							for(int zb = 0; zb < coordPrev_y+2*padding; zb++){
								if(za<padding || za>coordPrev_x+padding-1){
									pad_row_index.push_back(-1);
								}
								else if(zb<padding || zb>coordPrev_y+padding-1){
									pad_row_index.push_back(-1);
								}
								else{
									pad_row_index.push_back(zb + (za-padding)*coordPrev_y-padding + zc*coordPrev_x*coordPrev_y);
								}
							}
							pad_col_index.push_back(pad_row_index);
							pad_row_index.clear();
						}
						pad_index_matrix.push_back(pad_col_index);
						pad_col_index.clear();
					}
					//cout << coord_x <<"|"<< coord_y << endl;
					for(int aa =0; aa< coord_x; aa++) //y方向上的stride
					{
						for(int bb =0; bb < coord_y; bb++)  //x方向上的stride
						{
							for(int cc =0; cc< kernel_z; cc++)
							{
								for(int dd =0; dd< kernel_x; dd++)
								{
									for(int ee =0; ee<kernel_y; ee++)
									{
										temp_cell.push_back(pad_index_matrix[cc][aa*stride+dd][bb*stride+ee]);
									}
								}
							}
							temp_matrix.push_back(temp_cell);
							temp_cell.clear();
						}
					}
					all_conv_coord[wr].push_back(temp_matrix);
					temp_matrix.clear();
					pad_index_matrix.clear();
				}		
				//here
				/*
				if(padding == 0){
					for(int aa =0; aa< coord_x; aa++)
					{
						for(int bb =0; bb < coord_y; bb++)
						{
							for(int cc =0; cc< kernel_z; cc++)
							{
								for(int dd =0; dd< kernel_x; dd++)
								{
									for(int ee =0; ee<kernel_y; ee++)
									{
										temp_cell.push_back((aa+dd)*coordPrev_y + (ee+ bb) + cc*coordPrev_x*coordPrev_y);
									}
								}
							}
							temp_matrix.push_back(temp_cell);
							temp_cell.clear();
						}
					}
				all_conv_coord[wr].push_back(temp_matrix);
				temp_matrix.clear();
				}else if(padding)
				{
					for(int zc = 0; zc < kernel_z; zc++){
						for(int za = 0;za < coordPrev_x+2*padding; za++){
							for(int zb = 0; zb < coordPrev_y+2*padding; zb++){
								if(za<padding || za>coordPrev_x+padding-1){
									pad_row_index.push_back(-1);
								}
								else if(zb<padding || zb>coordPrev_y+padding-1){
									pad_row_index.push_back(-1);
								}
								else{
									pad_row_index.push_back(zb + (za-padding)*coordPrev_y-padding + zc*coordPrev_x*coordPrev_y);
								}
							}
							pad_col_index.push_back(pad_row_index);
							pad_row_index.clear();
						}
						pad_index_matrix.push_back(pad_col_index);
						pad_col_index.clear();
					}
					//cout << coord_x <<"|"<< coord_y << endl;
					for(int aa =0; aa< coord_x; aa++) //y方向上的stride
					{
						for(int bb =0; bb < coord_y; bb++)  //x方向上的stride
						{
							for(int cc =0; cc< kernel_z; cc++)
							{
								for(int dd =0; dd< kernel_x; dd++)
								{
									for(int ee =0; ee<kernel_y; ee++)
									{
										temp_cell.push_back(pad_index_matrix[cc][aa+dd][bb+ee]);
									}
								}
							}
							temp_matrix.push_back(temp_cell);
							temp_cell.clear();
						}
					}
					all_conv_coord[wr].push_back(temp_matrix);
					temp_matrix.clear();
					pad_index_matrix.clear();
				}		*/
				
			}
			else if(all_leyer_type[wr][ab] == 'p')
			{
				int coord_x =all_leyer_size[wr][ab][1];
				int coord_y=all_leyer_size[wr][ab][2];
				int coordPrev_x = all_leyer_size[wr][ab-1][1];
				int coordPrev_y = all_leyer_size[wr][ab-1][2];
				int kernel_x = all_leyer_size[wr][ab][4];
				int kernel_y= all_leyer_size[wr][ab][5];
				int stride = all_leyer_size[wr][ab][6];
				int horizontal =0;
				int vertical =0;
				temp_matrix.clear();
				temp_cell.clear();

				for(int aa =0; aa< coord_x; aa++)
				{
					for(int bb =0; bb<coord_y; bb++)
					{
						if(aa >0){vertical =1;}
						if(bb >0){horizontal =1;}
						if(horizontal ==0 && vertical ==0) //0,0
						{
							for(int cc=0; cc<kernel_x; cc++)
							{
								for(int dd =0; dd< kernel_y; dd++)
								{
									temp_cell.push_back((aa+cc)*coordPrev_y+(bb+dd));
								}
							}

						}else if(horizontal ==0 && vertical ==1)
						{
							for(int cc=0; cc<kernel_x; cc++)
							{
								for(int dd =0; dd< kernel_y; dd++)
								{
									temp_cell.push_back((aa*stride+cc)*coordPrev_y+(bb+dd));
								}
							}										
						}else if(horizontal ==1 && vertical == 0)
						{
							for(int cc=0; cc<kernel_x; cc++)
							{
								for(int dd =0; dd< kernel_y; dd++)
								{
									temp_cell.push_back((aa+cc)*coordPrev_y+(bb*stride+dd));
								}
							}
						}else if(horizontal ==1 && vertical == 1)
						{
							for(int cc=0; cc<kernel_x; cc++)
							{
								for(int dd =0; dd< kernel_y; dd++)
								{
									temp_cell.push_back((aa*stride+cc)*coordPrev_y+(bb*stride+dd));
								}
							}	
						}
						horizontal =0;
						vertical =0;
						temp_matrix.push_back(temp_cell);
						temp_cell.clear();
					}
				}
				all_pool_coord[wr].push_back(temp_matrix);
				temp_matrix.clear();
			}
		}
	}

	cout<<"Convolution and Pooling layer's related activities are completed."<<endl;
	cout <<  "size: "<<all_leyer_size[0].size()-1<<endl;
	cout <<  "size: "<<all_leyer_size[1].size()-1<<endl;
	/*--------------------Debugging-----------------------*/
	//cout<<"Conv deque Size: "<<all_conv_coord[wr].size()<<" Size zero: "<<all_conv_coord[wr][0].size()<<"Size One: "<<all_conv_coord[wr][1].size() <<endl;
	/*for(int gg =0; gg< all_conv_coord[wr][1][0].size(); gg++)
	{
		cout<<all_conv_coord[wr][1][0][gg]<<"--";
	}
	cout<<all_conv_coord[wr][1][0].size()<<endl;*/
	//cout<<"Pool deque Size: "<<all_pool_coord[wr].size()<<" Size zero: "<<all_pool_coord[wr][0].size()<<"Size One: "<<all_pool_coord[wr][1].size() <<endl;
	/*for(int gg=0; gg<all_pool_coord[wr][1][0].size(); gg++)
	{
		cout<<all_pool_coord[wr][1][0][gg]<<"--";
	}*/
	//cout<<"All pool Zero: "<< all_pool_coord[wr][0][0][0]<<endl;
	//cout<<"size: "<< all_pool_coord[wr][0][0].size()<<endl;
	/*----------------------------------------------------*/

    return true;
}



