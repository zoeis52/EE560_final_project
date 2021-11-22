function mrk = Converting_Arrow( mrko, varargin )
%% Reaching
% stimDef= {'S 11',    'S 21',      'S 31',  'S 41',   'S 51',  'S 61' 'S  8';
%          'Forward', 'Backward', 'Left',  'Right',  'Up',    'Down' 'Rest'};
      
%% Multi-Grasping
% stimDef= {'S 11', 'S 21', 'S 61', 'S  8';
%          'Cylindrical', 'Spherical', 'Lumbrical', 'Rest'};

%% Twisting
  stimDef= {'S 91',  'S101' 'S  8';
            'Twist_Left',  'Twist_Right' 'Rest' };

%% Default setting
miscDef= {'S 13',    'S 14';
          'Start',   'End'};

opt= propertylist2struct(varargin{:});
opt= set_defaults(opt, 'stimDef', stimDef, ...
                       'miscDef', miscDef);

mrk= mrk_defineClasses(mrko, opt.stimDef);
mrk.misc= mrk_defineClasses(mrko, opt.miscDef);