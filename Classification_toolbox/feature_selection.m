function fig = feature_selection()
% This is the machine-generated representation of a Handle Graphics object
% and its children.  Note that handle values may change when these objects
% are re-created. This may cause problems with any callbacks written to
% depend on the value of the handle at the time the object was saved.
% This problem is solved by saving the output as a FIG-file.
%
% To reopen this object, just type the name of the M-file at the MATLAB
% prompt. The M-file and its associated MAT-file must be on your path.
% 
% NOTE: certain newer features in MATLAB may not have been saved in this
% M-file due to limitations of this format, which has been superseded by
% FIG-files.  Figures which have been annotated using the plot editor tools
% are incompatible with the M-file/MAT-file format, and should be saved as
% FIG-files.

load feature_selection

v = version;

if (v(1) == '5') %Matlab 5
	h0 = figure('Units','characters', ...
        'Color',[0.8 0.8 0.8], ...
		'Colormap',mat0, ...
		'PaperPosition',[36 30 68 17], ...
		'PaperUnits','points', ...
        'Position',[36 30 68 20], ...
        'MenuBar','none', ...      
		'Tag','feature_selection');
else
    h0 = figure('Units','characters', ...
        'Color',[0.8 0.8 0.8], ...
		'Colormap',mat0, ...
		'FileName','C:\Work\classification_toolbox\feature_selection.m', ...
		'PaperPosition',[36 30 68 17], ...
		'PaperUnits','points', ...
        'Position',[36 30 68 20], ...
        'MenuBar','none', ...      
		'Tag','feature_selection', ...
		'ToolBar','none');
end
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'FontSize',18, ...
	'ListboxTop',0, ...
	'Position',[5 16.5 60 2], ...
	'String','Feature Selection', ...
	'Style','text', ...
	'Tag','StaticText1');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 14 60 1], ...
	'String','The data set you have selected has more than two dimensions', ...
	'Style','text', ...
	'Tag','StaticText2');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 12 60 1], ...
	'String','Please select a feature selection method to reduce the data to', ...
	'Style','text', ...
	'Tag','StaticText2');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 10 20 1], ...
	'String','two dimensions:', ...
	'Style','text', ...
	'Tag','StaticText2');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.87843137254902 0.815686274509804 0.596078431372549], ...
	'Callback','feature_selection_commands(''Changed_method'');', ...
	'ListboxTop',0, ...
	'Position',[34 9.75 30 1.5], ...
	'String',['PCA            ';'HDR            ';'Genetic_Culling'], ...
	'Style','popupmenu', ...
	'Tag','popMethod', ...
	'Value',1);
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 8 42 1], ...
	'String','Method parameters:', ...
    'FontWeight','Bold', ...
	'Style','text', ...
	'Tag','lblCap');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.8 0.8 0.8], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 6 42 1], ...
	'String','Out dimension:', ...
	'Style','text', ...
	'Tag','lblParameters');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[1 1 1], ...
	'HorizontalAlignment','left', ...
	'ListboxTop',0, ...
	'Position',[5 3.75 42 1.5], ...
	'String','2', ...
	'Style','edit', ...
	'Tag','txtParameters');
h1 = uicontrol('Parent',h0, ...
	'Units','characters', ...
	'BackgroundColor',[0.87843137254902 0.815686274509804 0.596078431372549], ...
	'Callback','feature_selection_commands(''OK'');', ...
	'ListboxTop',0, ...
	'Position',[55 1 10 3], ...
	'String','OK', ...
	'Tag','cmdOK');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'BackgroundColor',[1 1 1], ...
	'ListboxTop',0, ...
	'Position',[21 13.5 26.25 9], ...
	'Style','edit', ...
	'Tag','txtHiddenMethod', ...
	'Visible','off');
h1 = uicontrol('Parent',h0, ...
	'Units','points', ...
	'BackgroundColor',[1 1 1], ...
	'ListboxTop',0, ...
	'Position',[52.5 12.75 23.25 13.5], ...
	'Style','edit', ...
	'Tag','txtHiddenParams', ...
	'Visible','off');
feature_selection_commands('Init')
if nargout > 0, fig = h0; end
