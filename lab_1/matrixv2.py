# -*- coding: utf-8 -*-
# Python 3
import os
import traceback
from typing import Union, Dict, Tuple
import threading

import xbmc
import xbmcgui
import xbmcaddon

from xbmcaddon import Addon
from xbmc import LOGINFO as LOGNOTICE, LOGERROR, log

from resources.lib.handler.ParameterHandler import ParameterHandler
from resources.lib.handler.requestHandler import cRequestHandler
from resources.lib.handler.pluginHandler import cPluginHandler
from resources.lib.gui.guiElement import cGuiElement
from resources.lib.gui.gui import cGui
from resources.lib.gui.hoster import cHosterGui
from resources.lib.tmdbinfo import WindowsBoxes
from resources.lib.config import cConfig
from resources.lib.tools import logger
from resources.lib import updateManager
from resources.lib.tools import cPluginInfo, tools

PATH = xbmcaddon.Addon().getAddonInfo('path')
ART = os.path.join(PATH, 'resources', 'art')
LOGMESSAGE = cConfig().getLocalizedString(30166)
try:
    import resolveurl as resolver
except ImportError:
    # Resolver Fehlermeldung (bei defekten oder nicht installierten Resolver)
    xbmcgui.Dialog().ok(cConfig().getLocalizedString(30119), cConfig().getLocalizedString(30120))

def viewInfo(params: ParameterHandler) -> None:
    """
    Displays additional information about an item, such as metadata and year.

    Args:
        params (ParameterHandler): The handler containing parameters for the item to display.

    Returns:
        None
    """
    parms = ParameterHandler()
    sCleanTitle = params.getValue('searchTitle')
    sMeta = parms.getValue('sMeta')
    sYear = parms.getValue('sYear')
    WindowsBoxes(sCleanTitle, sCleanTitle, sMeta, sYear)


def parseUrl() -> None:
    """
    Parses the URL parameters and determines the action to perform based on the 'function' parameter.

    This function serves as the main entry point to various features, such as loading main menus,
    playing remote URLs, managing cache, viewing item info, and running global or alternative searches.

    Returns:
        None
    """
    params = ParameterHandler()
    logger.info(params.getAllParameters())
    # If no function is set, we set it to the default "load" function
    if params.exist('function'):
        sFunction = params.getValue('function')
        if sFunction == 'spacer':
            return True
        elif sFunction == 'clearCache':
            cRequestHandler('dummy').clearCache()
            return
        elif sFunction == 'viewInfo':
            viewInfo(params)
            return
        elif sFunction == 'searchAlternative':
            searchAlternative(params)
            return
        elif sFunction == 'searchTMDB':
            searchTMDB(params)
            return
        elif sFunction == 'devUpdates':
            updateManager.devUpdates()
            return
        elif sFunction == 'pluginInfo':
            cPluginInfo().pluginInfo()
            return
        elif sFunction == 'changelog':
            Addon().setSetting('changelog_version', '')
            tools.changelog()
            return
            
    elif params.exist('remoteplayurl'):
        try:
            remotePlayUrl = params.getValue('remoteplayurl')
            sLink = resolver.resolve(remotePlayUrl)
            if sLink:
                xbmc.executebuiltin('PlayMedia(' + sLink + ')')
            else:
                log(LOGMESSAGE + ' -> [MatrixV2]: Could not play remote url %s ' % sLink, LOGNOTICE)
        except resolver.resolver.ResolverError as e:
            log(LOGMESSAGE + ' -> [MatrixV2]: ResolverError: %s' % e, LOGERROR)
        return
    else:
        sFunction = 'load'

    # Test if we should run a function on a special site
    if not params.exist('site'):
        # As a default if no site was specified, we run the default starting gui with all plugins
        showMainMenu(sFunction)
        return
    sSiteName = params.getValue('site')
    if params.exist('playMode'):
        url = False
        playMode = params.getValue('playMode')
        isHoster = params.getValue('isHoster')
        url = params.getValue('url')
        manual = params.exist('manual')

        if cConfig().getSetting('hosterSelect') == 'Auto' and playMode != 'jd' and playMode != 'jd2' and playMode != 'pyload' and not manual:
            cHosterGui().streamAuto(playMode, sSiteName, sFunction)
        else:
            cHosterGui().stream(playMode, sSiteName, sFunction, url)
        return
    log(LOGMESSAGE + " -> [MatrixV2]: Call function '%s' from '%s'" % (sFunction, sSiteName), LOGNOTICE)
    # If the hoster gui is called, run the function on it and return
    if sSiteName == 'cHosterGui':
        showHosterGui(sFunction)
    # If global search is called
    elif sSiteName == 'globalSearch':
        searchterm = False
        if params.exist('searchterm'):
            searchterm = params.getValue('searchterm')
        searchGlobal(searchterm)
    elif sSiteName == 'Matrixv2':
        oGui = cGui()
        oGui.openSettings()
        oGui.updateDirectory()
    # Resolver Einstellungen im Hauptmenü
    elif sSiteName == 'resolver':
        resolver.display_settings()
    # Manuelles Update im Hauptmenü
    elif sSiteName == 'devUpdates':
        updateManager.devUpdates()
    # Plugin Infos    
    elif sSiteName == 'pluginInfo':
        cPluginInfo().pluginInfo()
    # Changelog anzeigen    
    elif sSiteName == 'changelog':
        tools.changelog()
    # Unterordner der Einstellungen   
    elif sSiteName == 'settings':
        oGui = cGui()
        for folder in settingsGuiElements():
            oGui.addFolder(folder)
        oGui.setEndOfDirectory()
    else:
        # Else load any other site as plugin and run the function
        plugin = __import__(sSiteName, globals(), locals())
        function = getattr(plugin, sFunction)
        function()


def showMainMenu(sFunction: str) -> None:
    """
    Displays the main menu by creating GUI elements for each plugin and additional settings.

    The function first adds a global search element if enabled in settings, then retrieves
    and displays all available plugins, sorted by ID. If no plugins are enabled, it opens the
    settings menu. If "Global Search Position" or "Settings Folder" settings are enabled, it
    adds respective elements to the GUI menu.

    Args:
        sFunction (str): The name of the function to be used when setting up each plugin’s GUI element.

    Returns:
        None
    """
    oGui = cGui()
    # Setzte die globale Suche an erste Stelle
    if cConfig().getSetting('GlobalSearchPosition') == 'true':
        oGui.addFolder(globalSearchGuiElement())
    oPluginHandler = cPluginHandler()
    aPlugins = oPluginHandler.getAvailablePlugins()
    if not aPlugins:
        log(LOGMESSAGE + ' -> [MatrixV2]: No activated Plugins found', LOGNOTICE)
        # Open the settings dialog to choose a plugin that could be enabled
        oGui.openSettings()
        oGui.updateDirectory()
    else:
        # Create a gui element for every plugin found
        for aPlugin in sorted(aPlugins, key=lambda k: k['id']):
            oGuiElement = cGuiElement()
            oGuiElement.setTitle(aPlugin['name'])
            oGuiElement.setSiteName(aPlugin['id'])
            oGuiElement.setFunction(sFunction)
            if 'icon' in aPlugin and aPlugin['icon']:
                oGuiElement.setThumbnail(aPlugin['icon'])
            oGui.addFolder(oGuiElement)
        if cConfig().getSetting('GlobalSearchPosition') == 'false':
            oGui.addFolder(globalSearchGuiElement())

    if cConfig().getSetting('SettingsFolder') == 'true':
        # Einstellung im Menü mit Untereinstellungen
        oGuiElement = cGuiElement()
        oGuiElement.setTitle(cConfig().getLocalizedString(30041))
        oGuiElement.setSiteName('settings')
        oGuiElement.setFunction('showSettingsFolder')
        oGuiElement.setThumbnail(os.path.join(ART, 'settings.png'))
        oGui.addFolder(oGuiElement)
    else:
        for folder in settingsGuiElements():
            oGui.addFolder(folder)
    oGui.setEndOfDirectory()


def settingsGuiElements() -> Tuple[cGuiElement, cGuiElement, cGuiElement, cGuiElement]:
    """
    Creates GUI elements for settings categories like plugin information, main settings, 
    resolver settings, and developer update manager.

    Each element is configured with a title, function, and thumbnail icon, returning a tuple of
    elements to be added to the GUI.

    Returns:
        Tuple[cGuiElement, cGuiElement, cGuiElement, cGuiElement]: A tuple containing GUI elements
        for plugin info, main settings, resolver settings, and developer update manager.
    """

    # GUI Plugin Informationen
    oGuiElement = cGuiElement()
    oGuiElement.setTitle(cConfig().getLocalizedString(30267))
    oGuiElement.setSiteName('pluginInfo')
    oGuiElement.setFunction('pluginInfo')
    oGuiElement.setThumbnail(os.path.join(ART, 'plugin_info.png'))
    PluginInfo = oGuiElement


    
    oGuiElement = cGuiElement()
    oGuiElement.setTitle(cConfig().getLocalizedString(30042))
    oGuiElement.setSiteName('Matrixv2')
    oGuiElement.setFunction('display_settings')
    oGuiElement.setThumbnail(os.path.join(ART, 'matrixv2_settings.png'))
    Matrixv2Settings = oGuiElement

    # GUI Resolver Einstellungen
    oGuiElement = cGuiElement()
    oGuiElement.setTitle(cConfig().getLocalizedString(30043))
    oGuiElement.setSiteName('resolver')
    oGuiElement.setFunction('display_settings')
    oGuiElement.setThumbnail(os.path.join(ART, 'resolveurl_settings.png'))
    resolveurlSettings = oGuiElement
    
    # GUI Nightly Updatemanager
    oGuiElement = cGuiElement()
    oGuiElement.setTitle(cConfig().getLocalizedString(30121))
    oGuiElement.setSiteName('devUpdates')
    oGuiElement.setFunction('devUpdates')
    oGuiElement.setThumbnail(os.path.join(ART, 'manuel_update.png'))
    DevUpdateMan = oGuiElement 
    return PluginInfo, Matrixv2Settings, resolveurlSettings, DevUpdateMan


def globalSearchGuiElement() -> cGuiElement:
    """
    Creates a GUI element for the global search functionality.

    This function initializes a GUI element specifically for the global search feature,
    setting its title, site name, function, and thumbnail image.

    Returns:
        cGuiElement: A configured GUI element for global search.
    """
    oGuiElement = cGuiElement()
    oGuiElement.setTitle(cConfig().getLocalizedString(30040))
    oGuiElement.setSiteName('globalSearch')
    oGuiElement.setFunction('globalSearch')
    oGuiElement.setThumbnail(os.path.join(ART, 'search.png'))
    return oGuiElement


def showHosterGui(sFunction: str) -> bool:
    """
    Displays the hoster GUI and calls a specific function.

    This function retrieves the hoster GUI and calls the specified function from it.

    Args:
        sFunction (str): The name of the function to be called on the hoster GUI.

    Returns:
        bool: Always returns True after executing the function.
    """
    oHosterGui = cHosterGui()
    function = getattr(oHosterGui, sFunction)
    function()
    return True


def searchGlobal(sSearchText: Union[str, bool] = False) -> bool:
    """
    Performs a global search using available plugins.

    If no search text is provided, it prompts the user for input. The function retrieves
    available plugins that support global search and executes a search in parallel threads.
    The results are collected and displayed in the GUI.

    Args:
        sSearchText (Union[str, bool], optional): The text to search for. Defaults to False.

    Returns:
        bool: Always returns True after executing the search.
    """
    oGui = cGui()
    oGui.globalSearch = True
    oGui._collectMode = True
    if not sSearchText:
        sSearchText = oGui.showKeyBoard()
    if not sSearchText: return True
    aPlugins = cPluginHandler().getAvailablePlugins()
    dialog = xbmcgui.DialogProgress()
    dialog.create(cConfig().getLocalizedString(30122), cConfig().getLocalizedString(30123))
    numPlugins = len(aPlugins)
    threads = []
    for count, pluginEntry in enumerate(aPlugins):
        if pluginEntry['globalsearch'] == 'false':
            continue
        dialog.update((count + 1) * 50 // numPlugins, cConfig().getLocalizedString(30124) + str(pluginEntry['name']) + '...')
        if dialog.iscanceled(): return
        log(LOGMESSAGE + ' -> [MatrixV2]: Searching for %s at %s' % (sSearchText, pluginEntry['id']), LOGNOTICE)

        t = threading.Thread(target=_pluginSearch, args=(pluginEntry, sSearchText, oGui), name=pluginEntry['name'])
        threads += [t]
        t.start()
    for count, t in enumerate(threads):
        if dialog.iscanceled(): return
        t.join()
        dialog.update((count + 1) * 50 // numPlugins + 50, t.getName() + cConfig().getLocalizedString(30125))
    dialog.close()
    # deactivate collectMode attribute because now we want the elements really added
    oGui._collectMode = False
    total = len(oGui.searchResults)
    dialog = xbmcgui.DialogProgress()
    dialog.create(cConfig().getLocalizedString(30126), cConfig().getLocalizedString(30127))
    for count, result in enumerate(sorted(oGui.searchResults, key = lambda k: k['guiElement'].getSiteName()), 1):
        if dialog.iscanceled(): return
        oGui.addFolder(result['guiElement'], result['params'], bIsFolder=result['isFolder'], iTotal=total)
        dialog.update(count * 100 // total, str(count) + cConfig().getLocalizedString(30128) + str(total) + ': ' + result['guiElement'].getTitle())
    dialog.close()
    oGui.setView()
    oGui.setEndOfDirectory()
    return True


def searchAlternative(params: ParameterHandler) -> bool:
    """
    Searches for alternative media options based on provided parameters.

    This function retrieves the search title, IMDb ID, and year from the parameters,
    then executes a search using available plugins. The results are filtered based on
    the search criteria and displayed in the GUI.

    Args:
        params (ParameterHandler): The parameters containing search details.

    Returns:
        bool: Always returns True after executing the search.
    """
    searchTitle = params.getValue('searchTitle')
    searchImdbId = params.getValue('searchImdbID')
    searchYear = params.getValue('searchYear')
    oGui = cGui()
    oGui.globalSearch = True
    oGui._collectMode = True
    aPlugins = []
    aPlugins = cPluginHandler().getAvailablePlugins()
    dialog = xbmcgui.DialogProgress()
    dialog.create(cConfig().getLocalizedString(30122), cConfig().getLocalizedString(30123))
    numPlugins = len(aPlugins)
    threads = []
    for count, pluginEntry in enumerate(aPlugins):
        if dialog.iscanceled(): return
        dialog.update((count + 1) * 50 // numPlugins, cConfig().getLocalizedString(30124) + str(pluginEntry['name']) + '...')
        log(LOGMESSAGE + ' -> [MatrixV2]: Searching for ' + searchTitle + pluginEntry['id'], LOGNOTICE)
        t = threading.Thread(target=_pluginSearch, args=(pluginEntry, searchTitle, oGui), name=pluginEntry['name'])
        threads += [t]
        t.start()
    for count, t in enumerate(threads):
        t.join()
        if dialog.iscanceled(): return
        dialog.update((count + 1) * 50 // numPlugins + 50, t.getName() + cConfig().getLocalizedString(30125))
    dialog.close()
    # check results, put this to the threaded part, too
    filteredResults = []
    for result in oGui.searchResults:
        guiElement = result['guiElement']
        log(LOGMESSAGE + ' -> [MatrixV2]: Site: %s Titel: %s' % (guiElement.getSiteName(), guiElement.getTitle()), LOGNOTICE)
        if searchTitle not in guiElement.getTitle():
            continue
        if guiElement._sYear and searchYear and guiElement._sYear != searchYear: continue
        if searchImdbId and guiElement.getItemProperties().get('imdbID', False) and guiElement.getItemProperties().get('imdbID', False) != searchImdbId: continue
        filteredResults.append(result)
    oGui._collectMode = False
    total = len(filteredResults)
    for result in sorted(filteredResults, key=lambda k: k['guiElement'].getSiteName()):
        oGui.addFolder(result['guiElement'], result['params'], bIsFolder=result['isFolder'], iTotal=total)
    oGui.setView()
    oGui.setEndOfDirectory()
    xbmc.executebuiltin('Container.Update')
    return True


def searchTMDB(params: ParameterHandler) -> bool:
    """
    Searches for media using The Movie Database (TMDB) based on provided parameters.

    This function retrieves the search title from the parameters and executes a search
    using available plugins that support global search. The results are displayed in the GUI.

    Args:
        params (ParameterHandler): The parameters containing search details.

    Returns:
        bool: Always returns True after executing the search.
    """
    sSearchText = params.getValue('searchTitle')
    oGui = cGui()
    oGui.globalSearch = True
    oGui._collectMode = True
    if not sSearchText: return True
    aPlugins = []
    aPlugins = cPluginHandler().getAvailablePlugins()
    dialog = xbmcgui.DialogProgress()
    dialog.create(cConfig().getLocalizedString(30122), cConfig().getLocalizedString(30123))
    numPlugins = len(aPlugins)
    threads = []
    for count, pluginEntry in enumerate(aPlugins):
        if pluginEntry['globalsearch'] == 'false':
            continue
        if dialog.iscanceled(): return
        dialog.update((count + 1) * 50 // numPlugins, cConfig().getLocalizedString(30124) + str(pluginEntry['name']) + '...')
        log(LOGMESSAGE + ' -> [MatrixV2]: Searching for %s at %s' % (sSearchText, pluginEntry['id']), LOGNOTICE)

        t = threading.Thread(target = _pluginSearch, args = (pluginEntry, sSearchText, oGui), name = pluginEntry['name'])
        threads += [t]
        t.start()
    for count, t in enumerate(threads):
        t.join()
        if dialog.iscanceled(): return
        dialog.update((count + 1) * 50 // numPlugins + 50, t.getName() + cConfig().getLocalizedString(30125))
    dialog.close()
    # deactivate collectMode attribute because now we want the elements really added
    oGui._collectMode = False
    total = len(oGui.searchResults)
    dialog = xbmcgui.DialogProgress()
    dialog.create(cConfig().getLocalizedString(30126), cConfig().getLocalizedString(30127))
    for count, result in enumerate(sorted(oGui.searchResults, key=lambda k: k['guiElement'].getSiteName()), 1):
        if dialog.iscanceled(): return
        oGui.addFolder(result['guiElement'], result['params'], bIsFolder=result['isFolder'], iTotal=total)
        dialog.update(count * 100 // total, str(count) + cConfig().getLocalizedString(30128) + str(total) + ': ' + result['guiElement'].getTitle())
    dialog.close()
    oGui.setView()
    oGui.setEndOfDirectory()
    return True


def _pluginSearch(pluginEntry: Dict[str, str], sSearchText: str, oGui: cGuiElement) -> None:
    """
    Executes a search using a specific plugin.

    This function imports the specified plugin and calls its search function, handling
    any exceptions that may occur during the search.

    Args:
        pluginEntry (Dict[str, str]): The entry for the plugin containing its ID and other details.
        sSearchText (str): The search text to be used.
        oGui (cGuiElement): The GUI element to display results.

    Returns:
        None
    """
    try:
        plugin = __import__(pluginEntry['id'], globals(), locals())
        function = getattr(plugin, '_search')
        function(oGui, sSearchText)
    except Exception:
        log(LOGMESSAGE + ' -> [MatrixV2]: ' + pluginEntry['name'] + ': search failed', LOGERROR)
        log(traceback.format_exc())