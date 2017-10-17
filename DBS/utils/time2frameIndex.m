function [f1, f2]=time2frameIndex(t1,t2, duration, f_all)

tt1 = min(t1,t2);
tt2 = max(t1,t2);

f1 = max(1,round(f_all*tt1/duration));
f2 = min(f_all,round(f_all*tt2/duration));

end