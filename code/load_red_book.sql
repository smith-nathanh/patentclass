#
# Creates the mysql database the patentclass project
# and loads the csv files generated by the get_red_book.py
# to create two main tables:
#     ipc - for each patent publication, identify the ipc components (section, class, subclass, group, subgroup)
#     txt - for each patent publication, identify the textual material (abstract, description, claims)


#
# create the database
#

create database patentclass;
use patentclass;

#
# Load the patent IPC code components into the ipc table
#

create or replace table ipc (wk CHAR(4),
                  pub VARCHAR(20),
                  tri VARCHAR(20),
                  ver VARCHAR(20),
                  lev VARCHAR(20),
                  sec VARCHAR(20),
                  cla VARCHAR(20),
                  subc VARCHAR(20),
                  grp VARCHAR(20),
                  subg VARCHAR(20),
                  pos VARCHAR(20),
                  cv VARCHAR(20),
                  act VARCHAR(20),
                  gen VARCHAR(20),
                  sta VARCHAR(20),
                  src VARCHAR(20));

load data local infile 'D:/Projects/ipg/ipg181225/ipc181225.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1225';
load data local infile 'D:/Projects/ipg/ipg181218/ipc181218.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1218';
load data local infile 'D:/Projects/ipg/ipg181211/ipc181211.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1211';
load data local infile 'D:/Projects/ipg/ipg181204/ipc181204.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1204';
load data local infile 'D:/Projects/ipg/ipg181127/ipc181127.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1127';
load data local infile 'D:/Projects/ipg/ipg181120/ipc181120.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1120';
load data local infile 'D:/Projects/ipg/ipg181113/ipc181113.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1113';
load data local infile 'D:/Projects/ipg/ipg181106/ipc181106.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1106';
load data local infile 'D:/Projects/ipg/ipg181030/ipc181030.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1030';
load data local infile 'D:/Projects/ipg/ipg181023/ipc181023.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1023';
load data local infile 'D:/Projects/ipg/ipg181016/ipc181016.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1016';
load data local infile 'D:/Projects/ipg/ipg181009/ipc181009.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1009';
load data local infile 'D:/Projects/ipg/ipg181002/ipc181002.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '1002';
load data local infile 'D:/Projects/ipg/ipg180925/ipc180925.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0925';
load data local infile 'D:/Projects/ipg/ipg180918/ipc180918.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0918';
load data local infile 'D:/Projects/ipg/ipg180911/ipc180911.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0911';
load data local infile 'D:/Projects/ipg/ipg180904/ipc180904.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0904';
load data local infile 'D:/Projects/ipg/ipg180828/ipc180828.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0828';
load data local infile 'D:/Projects/ipg/ipg180821/ipc180821.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0821';
load data local infile 'D:/Projects/ipg/ipg180814/ipc180814.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0814';
load data local infile 'D:/Projects/ipg/ipg180807/ipc180807.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0807';
load data local infile 'D:/Projects/ipg/ipg180731/ipc180731.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0731';
load data local infile 'D:/Projects/ipg/ipg180724/ipc180724.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0724';
load data local infile 'D:/Projects/ipg/ipg180717/ipc180717.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0717';
load data local infile 'D:/Projects/ipg/ipg180710/ipc180710.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0710';
load data local infile 'D:/Projects/ipg/ipg180703/ipc180703.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0703';
load data local infile 'D:/Projects/ipg/ipg180626/ipc180626.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0626';
load data local infile 'D:/Projects/ipg/ipg180619/ipc180619.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0619';
load data local infile 'D:/Projects/ipg/ipg180612/ipc180612.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0612';
load data local infile 'D:/Projects/ipg/ipg180605/ipc180605.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0605';
load data local infile 'D:/Projects/ipg/ipg180529/ipc180529.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0529';
load data local infile 'D:/Projects/ipg/ipg180522/ipc180522.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0522';
load data local infile 'D:/Projects/ipg/ipg180515/ipc180515.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0515';
load data local infile 'D:/Projects/ipg/ipg180508/ipc180508.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0508';
load data local infile 'D:/Projects/ipg/ipg180501/ipc180501.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0501';
load data local infile 'D:/Projects/ipg/ipg180424/ipc180424.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0424';
load data local infile 'D:/Projects/ipg/ipg180417/ipc180417.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0417';
load data local infile 'D:/Projects/ipg/ipg180410/ipc180410.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0410';
load data local infile 'D:/Projects/ipg/ipg180403/ipc180403.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0403';
load data local infile 'D:/Projects/ipg/ipg180327/ipc180327.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0327';
load data local infile 'D:/Projects/ipg/ipg180320/ipc180320.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0320';
load data local infile 'D:/Projects/ipg/ipg180313/ipc180313.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0313';
load data local infile 'D:/Projects/ipg/ipg180306/ipc180306.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0306';
load data local infile 'D:/Projects/ipg/ipg180227/ipc180227.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0227';
load data local infile 'D:/Projects/ipg/ipg180220/ipc180220.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0220';
load data local infile 'D:/Projects/ipg/ipg180213/ipc180213.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0213';
load data local infile 'D:/Projects/ipg/ipg180206/ipc180206.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0206';
load data local infile 'D:/Projects/ipg/ipg180130/ipc180130.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0130';
load data local infile 'D:/Projects/ipg/ipg180123/ipc180123.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0123';
load data local infile 'D:/Projects/ipg/ipg180116/ipc180116.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0116';
load data local infile 'D:/Projects/ipg/ipg180109/ipc180109.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0109';
load data local infile 'D:/Projects/ipg/ipg180102/ipc180102.csv' into table ipc fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,tri,ver,lev,sec,cla,subc,grp,subg,pos,cv,act,gen,sta,src) set wk  = '0102';
create or replace index ind_ipc_pub on ipc (pub);

#
# Load the patent text components into the txt table
#

create table txt (wk CHAR(4),
                  pub VARCHAR(20),
                  typ VARCHAR(20),
                  dsc LONGTEXT,
                  abstr LONGTEXT,
                  clms LONGTEXT);

load data local infile 'D:/Projects/ipg/ipg181225/txt181225.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1225'; 
load data local infile 'D:/Projects/ipg/ipg181218/txt181218.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1218'; 
load data local infile 'D:/Projects/ipg/ipg181211/txt181211.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1211'; 
load data local infile 'D:/Projects/ipg/ipg181204/txt181204.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1204'; 
load data local infile 'D:/Projects/ipg/ipg181127/txt181127.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1127'; 
load data local infile 'D:/Projects/ipg/ipg181120/txt181120.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1120'; 
load data local infile 'D:/Projects/ipg/ipg181113/txt181113.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1113'; 
load data local infile 'D:/Projects/ipg/ipg181106/txt181106.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1106'; 
load data local infile 'D:/Projects/ipg/ipg181030/txt181030.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1030'; 
load data local infile 'D:/Projects/ipg/ipg181023/txt181023.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1023'; 
load data local infile 'D:/Projects/ipg/ipg181016/txt181016.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1016'; 
load data local infile 'D:/Projects/ipg/ipg181009/txt181009.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1009'; 
load data local infile 'D:/Projects/ipg/ipg181002/txt181002.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '1002'; 
load data local infile 'D:/Projects/ipg/ipg180925/txt180925.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0925'; 
load data local infile 'D:/Projects/ipg/ipg180918/txt180918.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0918'; 
load data local infile 'D:/Projects/ipg/ipg180911/txt180911.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0911'; 
load data local infile 'D:/Projects/ipg/ipg180904/txt180904.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0904'; 
load data local infile 'D:/Projects/ipg/ipg180828/txt180828.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0828'; 
load data local infile 'D:/Projects/ipg/ipg180821/txt180821.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0821'; 
load data local infile 'D:/Projects/ipg/ipg180814/txt180814.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0814'; 
load data local infile 'D:/Projects/ipg/ipg180807/txt180807.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0807'; 
load data local infile 'D:/Projects/ipg/ipg180731/txt180731.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0731'; 
load data local infile 'D:/Projects/ipg/ipg180724/txt180724.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0724'; 
load data local infile 'D:/Projects/ipg/ipg180717/txt180717.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0717'; 
load data local infile 'D:/Projects/ipg/ipg180710/txt180710.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0710'; 
load data local infile 'D:/Projects/ipg/ipg180703/txt180703.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0703'; 
load data local infile 'D:/Projects/ipg/ipg180626/txt180626.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0626'; 
load data local infile 'D:/Projects/ipg/ipg180619/txt180619.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0619'; 
load data local infile 'D:/Projects/ipg/ipg180612/txt180612.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0612'; 
load data local infile 'D:/Projects/ipg/ipg180605/txt180605.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0605'; 
load data local infile 'D:/Projects/ipg/ipg180529/txt180529.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0529'; 
load data local infile 'D:/Projects/ipg/ipg180522/txt180522.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0522'; 
load data local infile 'D:/Projects/ipg/ipg180515/txt180515.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0515'; 
load data local infile 'D:/Projects/ipg/ipg180508/txt180508.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0508'; 
load data local infile 'D:/Projects/ipg/ipg180501/txt180501.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0501'; 
load data local infile 'D:/Projects/ipg/ipg180424/txt180424.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0424'; 
load data local infile 'D:/Projects/ipg/ipg180417/txt180417.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0417'; 
load data local infile 'D:/Projects/ipg/ipg180410/txt180410.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0410'; 
load data local infile 'D:/Projects/ipg/ipg180403/txt180403.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0403'; 
load data local infile 'D:/Projects/ipg/ipg180327/txt180327.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0327'; 
load data local infile 'D:/Projects/ipg/ipg180320/txt180320.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0320'; 
load data local infile 'D:/Projects/ipg/ipg180313/txt180313.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0313'; 
load data local infile 'D:/Projects/ipg/ipg180306/txt180306.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0306'; 
load data local infile 'D:/Projects/ipg/ipg180227/txt180227.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0227'; 
load data local infile 'D:/Projects/ipg/ipg180220/txt180220.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0220'; 
load data local infile 'D:/Projects/ipg/ipg180213/txt180213.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0213'; 
load data local infile 'D:/Projects/ipg/ipg180206/txt180206.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0206'; 
load data local infile 'D:/Projects/ipg/ipg180130/txt180130.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0130'; 
load data local infile 'D:/Projects/ipg/ipg180123/txt180123.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0123'; 
load data local infile 'D:/Projects/ipg/ipg180116/txt180116.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0116'; 
load data local infile 'D:/Projects/ipg/ipg180109/txt180109.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0109'; 
load data local infile 'D:/Projects/ipg/ipg180102/txt180102.csv' into table txt fields terminated by ',' lines terminated by '\n' ignore 1 rows (pub,typ,dsc,abstr,clms) set wk = '0102';
create or replace index ind_txt_pub on txt (pub);




